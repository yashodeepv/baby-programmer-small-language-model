"""curriculum.py - The book: IR2 flows taught one idea per chapter.

The premise of baby-programmer is that a programmer is not made by
web-scale compute. A person learns from one class, one good book, and then
the job. This module is the book. Each chapter adds a single new idea on top
of the earlier ones, every flow in it must use that idea, and the flows
grow a little longer and the expressions a little deeper as it goes.

Every exercise comes with its answer key already checked: each flow passes
the verifier and has been run until every edge was taken. The mock exams
are the held-out splits, per chapter: wording the model has never read
(style 2) and a whole business domain it has never seen (logistics).

    python3.11 -m flow.gen.curriculum            table of contents, one sample per chapter
"""

import random
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

from flow.gen.sampler import sample_flow


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    teaches: str
    patterns: Tuple[str, ...]           # allowed, cumulative
    require: Optional[str]              # the new idea every flow must use
    n_patterns: Tuple[int, int]
    depth: int                          # expression nesting


CHAPTERS = [
    Chapter(1, 'Hello, flow',
            'a trigger starts it, one action happens, it ends; values come from the request',
            ('notify', 'persist'), None, (1, 1), 0),
    Chapter(2, 'Working things out',
            'compute steps name new values from old ones',
            ('notify', 'persist', 'compute'), 'compute', (1, 2), 0),
    Chapter(3, 'Rules and rejections',
            'validate a rule; stop with an error, or fail the whole flow',
            ('notify', 'persist', 'compute', 'validate'), 'validate', (2, 3), 1),
    Chapter(4, 'Talking to other systems',
            'call a service, use its answer, retry, and handle failure',
            ('notify', 'persist', 'compute', 'validate', 'call'), 'call', (2, 3), 1),
    Chapter(5, 'Decisions',
            'branch on a condition; one path, the other, or a way out',
            ('notify', 'persist', 'compute', 'validate', 'call', 'branch'), 'branch', (2, 4), 1),
    Chapter(6, 'Doing things at once',
            'run branches in parallel and join them',
            ('notify', 'persist', 'compute', 'validate', 'call', 'branch', 'parallel'), 'parallel',
            (3, 5), 1),
    Chapter(7, 'Putting it together',
            'longer flows using everything, with deeper expressions',
            ('notify', 'persist', 'compute', 'validate', 'call', 'branch', 'parallel'), None,
            (4, 6), 2),
]


def chapter_flows(chapter, n, catalogs, seed=0, max_steps=16):
    """n flows for one chapter. Yields (flow, meta); meta carries 'chapter'."""
    rng = random.Random(seed * 1000 + chapter.number)
    names = sorted(catalogs)
    for i in range(n):
        cat = catalogs[rng.choice(names)]
        flow, meta = sample_flow(rng, cat, n_patterns=chapter.n_patterns, max_steps=max_steps,
                                 name=f'ch{chapter.number}_{cat.name}_{i}',
                                 patterns=chapter.patterns, require=chapter.require,
                                 depth=chapter.depth)
        meta['chapter'] = chapter.number
        yield flow, meta


def book(per_chapter, catalogs, seed=0, max_steps=16):
    """Every chapter in order."""
    for ch in CHAPTERS:
        yield from chapter_flows(ch, per_chapter, catalogs, seed, max_steps)


def main():
    from flow.gen.catalog import all_catalogs
    from flow.gen.render import render
    cats = all_catalogs()
    for ch in CHAPTERS:
        flow, meta = next(chapter_flows(ch, 1, cats, seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0))
        text, _ = render(flow, 0, meta['noun'])
        print(f'Chapter {ch.number}. {ch.title}')
        print(f'  teaches: {ch.teaches}')
        print(f'  e.g.  ({meta["n_steps"]} steps) {text}')
        print()


if __name__ == '__main__':
    main()
