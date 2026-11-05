"""Pretend a channel is connected, using a fake token, so the banner can be tested with no account."""

from clipforge.db import session_scope
from clipforge.models import Channel
from clipforge.uploads.base import save_tokens

with session_scope() as session:
    channel = Channel(name="Test channel", platform="youtube", connected=True, account_name="test")
    session.add(channel)
    session.commit()
    save_tokens(channel.id, {"access_token": "fake", "refresh_token": "fake", "expires_at": 9e12})
