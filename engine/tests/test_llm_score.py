import json
import sys
import textwrap

import pytest

from clipforge.analysis import llm_score
from clipforge.analysis.scoring import Moment
from clipforge.config import get_settings

FAKE_SERVER = textwrap.dedent(
    """\
    #!{python}
    # Stands in for llama-server: same flags, same two endpoints, a fixed rating.
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    assert args["--host"] == "127.0.0.1", "must only listen on loopback"
    assert "--offline" in sys.argv
    key = args["--api-key"]
    reply = {reply!r}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, {{"status": "ok"}})

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + key:
                return self.send(401, {{"error": "no key"}})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0.0
            assert body["response_format"]["type"] == "json_schema"
            self.send(200, {{"choices": [{{"message": {{"content": reply}}}}]}})

    HTTPServer(("127.0.0.1", int(args["--port"])), Handler).serve_forever()
    """
)


def install_fake(tmp_path, reply):
    folder = get_settings().work_dir / "tools" / "llama" / "build" / "bin"
    folder.mkdir(parents=True)
    server = folder / "llama-server"
    server.write_text(FAKE_SERVER.format(python=sys.executable, reply=reply))
    server.chmod(0o755)
    models = get_settings().models_dir / "llm"
    models.mkdir(parents=True)
    (models / llm_score.MODEL_FILE).write_bytes(b"not read by the fake")
    return server


def moments() -> list[Moment]:
    return [
        Moment(0.0, 10.0, 0.6, {}, "first clip text", "why"),
        Moment(20.0, 30.0, 0.5, {}, "second clip text", "why"),
        Moment(2.0, 12.0, 0.4, {}, "a near copy of the first", "why"),
    ]


def test_nothing_changes_when_the_model_is_missing():
    found, method = llm_score.rescore(moments())
    assert method == "editor"
    assert [m.score for m in found] == [0.6, 0.5, 0.4]


def test_ratings_blend_into_the_score_and_are_recorded(tmp_path):
    server = install_fake(tmp_path, json.dumps({"hook": 10, "standalone": 10, "payoff": 10, "interest": 10}))
    if not server.exists():
        pytest.skip("could not install the fake server")
    found, method = llm_score.rescore(moments())
    assert method == "local model"
    assert len(found) == 2  # The near copy is not rated.
    assert all(m.reasons["llm_payoff"] == 10 for m in found)
    assert found[0].score == pytest.approx(0.5 * 0.6 + 0.5 * 1.0)


def test_a_bad_reply_keeps_the_editors_ranking(tmp_path):
    install_fake(tmp_path, "this is not json")
    original = moments()
    found, method = llm_score.rescore(original)
    assert method == "editor"
    assert [m.score for m in found] == [0.6, 0.5, 0.4]
    assert all(not any(k.startswith("llm_") for k in m.reasons) for m in found)


def test_out_of_range_ratings_are_clamped(tmp_path):
    install_fake(tmp_path, json.dumps({"hook": 99, "standalone": -4, "payoff": 5, "interest": 5}))
    runtime, model = llm_score.runtime_path(), llm_score.model_path()
    with llm_score.LocalModel(model, runtime) as local:
        assert local.rate("text") == {"hook": 10, "standalone": 0, "payoff": 5, "interest": 5}


def test_the_server_is_stopped_afterwards(tmp_path):
    install_fake(tmp_path, json.dumps({"hook": 5, "standalone": 5, "payoff": 5, "interest": 5}))
    with llm_score.LocalModel(llm_score.model_path(), llm_score.runtime_path()) as local:
        process = local.process
    assert process is not None and process.poll() is not None


def test_cuda_runtime_and_libgomp_folders_are_on_the_library_path(tmp_path):
    root = tmp_path / "llama"
    program = root / "b1" / "llama-b1"
    cudart = root / "b1" / "cudart-llama-b1"
    empty = root / "b1" / "notes"
    for folder in (program, cudart, empty, root / "lib"):
        folder.mkdir(parents=True)
    (program / "llama-server").write_text("")
    (cudart / "libcudart.so.12").write_text("")
    assert llm_score.library_dirs(program / "llama-server") == [program, cudart, root / "lib"]


def test_a_server_that_fails_to_start_says_why(tmp_path):
    server = install_fake(tmp_path, "{}")
    server.write_text(
        f"#!{sys.executable}\nimport sys\nprint('error while loading libggml-cuda.so')\nsys.exit(3)\n"
    )
    with pytest.raises(RuntimeError, match="libggml-cuda"):
        with llm_score.LocalModel(llm_score.model_path(), llm_score.runtime_path()):
            pass


def test_failing_clips_are_never_rated():
    weak = moments()
    for moment in weak:
        moment.passed = False
    assert llm_score.rescore(weak) == (weak, "editor")


def test_shortlist_skips_near_copies():
    assert [m.text for m in llm_score.shortlist(moments())] == ["first clip text", "second clip text"]


def test_interest_and_payoff_count_double():
    assert llm_score.rating_score(
        {"hook": 0, "standalone": 0, "payoff": 10, "interest": 10}
    ) == pytest.approx(40 / 60)
