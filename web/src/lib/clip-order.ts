import type { Clip } from "@/lib/api";

/** Newest job first, as the API lists them, but a job's own clips in Part order (1, 2, 3). */
export function inPartOrder(clips: Clip[]): Clip[] {
  const jobRank = new Map<string, number>();
  clips.forEach((c) => {
    if (!jobRank.has(c.job_id)) jobRank.set(c.job_id, jobRank.size);
  });
  return [...clips].sort(
    (a, b) => jobRank.get(a.job_id)! - jobRank.get(b.job_id)! || (a.part ?? Infinity) - (b.part ?? Infinity),
  );
}
