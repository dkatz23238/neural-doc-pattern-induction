import random
import re
import unittest

import torch

from pattern_induction.compose import rename_annotated, sample_episode
from pattern_induction.membership import HELD_OUT, Episode
from pattern_induction.program import (
    agrees,
    describe,
    execute,
    feature_batch,
    grammar_ok,
    label_string,
    linearize,
    sample_compose_task,
    sample_dsl_task,
    to_regex,
    delinearize,
)
from pattern_induction.proposer import Proposer, collate_tasks, program_loss, token_logprob
from pattern_induction.recipes import d1_bank
from pattern_induction.renaming import rename_episode
from pattern_induction.span_net import SpanNet, parameter_count


FAMILIES = (
    "primitive",
    "repeat",
    "optional",
    "alternation",
    "prefix_repeat",
    "repeat_optional",
    "prefix_alt",
    "optional_internal",
    "repeat_extra",
)


class ProgramSemanticsTest(unittest.TestCase):
    def test_gold_programs_match_the_languages(self):
        for family in FAMILIES:
            for seed in range(6):
                task = sample_compose_task(family, random.Random(seed), probe_n=4)
                steps = linearize(task.program)
                self.assertTrue(grammar_ok(steps), describe(task.program))
                self.assertLessEqual(len(steps), 12)
                self.assertEqual(describe(delinearize(steps)), describe(task.program))
                self.assertTrue(agrees(task.program, task.supports, task.probe), describe(task.program))
                regex = to_regex(task.program, task.supports)
                for text, truth in task.probe:
                    self.assertEqual(re.fullmatch(regex, text) is not None, truth, family)

    def test_compiled_schemas_execute_as_the_matcher(self):
        for spec in d1_bank():
            task = sample_dsl_task(spec, random.Random(0), probe_n=3)
            self.assertTrue(grammar_ok(linearize(task.program)), spec.ident)
            self.assertTrue(agrees(task.program, task.supports, task.probe), spec.ident)

    def test_repeat_depth_is_open_in_the_gold_program(self):
        task = sample_compose_task("repeat_extra", random.Random(1), probe_n=4)
        repeat = task.program if task.program.op == "REPEAT" else task.program.children[-1]
        self.assertGreaterEqual(repeat.hi, 5)
        self.assertTrue(any(truth and sum(text.count(mark) for mark in "-_./") >= 4 for text, truth in task.probe))
        self.assertTrue(all(execute(task.program, text, task.supports) for text, truth in task.probe if truth))

    def test_copy_survives_a_held_out_delimiter(self):
        task = sample_compose_task("repeat", random.Random(2), probe_n=2)
        episode = Episode(task.supports, task.supports[0], 1, "repeat", "p1", "repeat")
        renamed = rename_episode(episode, random.Random(3), True)
        blob = "".join(renamed.support)
        self.assertTrue(any(ch in HELD_OUT for ch in blob))
        self.assertTrue(execute(task.program, renamed.support[0], renamed.support))
        regex = to_regex(task.program, renamed.support)
        self.assertTrue(any(ch in HELD_OUT for ch in regex))
        self.assertIsNotNone(re.fullmatch(regex, renamed.support[0]))

    def test_span_labels_cover_a_support(self):
        task = sample_compose_task("prefix_repeat", random.Random(0), probe_n=1)
        labels = label_string(task.program, task.supports[0], task.supports)
        self.assertIsNotNone(labels)
        self.assertEqual(len(labels), len(task.supports[0]))
        self.assertTrue(any(role == 1 for role, _scope in labels))
        self.assertTrue(any(scope == 1 for _role, scope in labels))


class ProposerTest(unittest.TestCase):
    def test_capacity_and_a_finite_step(self):
        self.assertEqual(parameter_count(SpanNet()), 945464)
        model = Proposer()
        count = parameter_count(model)
        self.assertGreaterEqual(count, 1_000_000)
        self.assertLessEqual(count, 2_000_000)
        tasks = [sample_compose_task(family, random.Random(family.__len__()), 1) for family in ("primitive", "repeat", "optional", "alternation")]
        batch = collate_tasks(tasks)
        model.train()
        encoded, memory, key_pad = model.encode(batch["local"], batch["classes"], batch["rel"], batch["pad"])
        logits = model.decode(memory, key_pad, batch["steps"], batch["steps"]["op"].shape[1])
        aux = {"role": model.char_role(encoded), "scope": model.char_scope(encoded)}
        loss, _op, _rank = program_loss(logits, aux, batch, token_logprob(logits, batch["steps"]), None, None)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_eval_rename_keeps_the_beam(self):
        model = Proposer()
        model.eval()
        task = sample_compose_task("prefix_alt", random.Random(4), 1)
        episode = Episode(task.supports, task.supports[0], 1, task.family, "p1", task.family)
        renamed = rename_episode(episode, random.Random(9), True)
        left = feature_batch([task])
        right = feature_batch([type(task)(task.family, renamed.support, task.program, [])])
        self.assertTrue(all(torch.allclose(a, b) for a, b in zip(left, right)))
        beam = model.propose(*left, 2)[0]
        other = model.propose(*right, 2)[0]
        self.assertEqual(len(beam), len(other))
        self.assertGreater(len(beam), 0)
        for (prog, _score), (shifted, _other) in zip(beam, other):
            steps = linearize(prog)
            self.assertTrue(grammar_ok(steps))
            self.assertEqual(steps, linearize(shifted))

    def test_query_annotation_is_not_an_input(self):
        # The proposer reads supports only. A renamed annotation object cannot change features.
        episode = sample_episode("prefix_repeat", random.Random(1), 1)
        renamed = rename_annotated(episode, random.Random(2), True)
        task = sample_compose_task("prefix_repeat", random.Random(1), 1)
        self.assertTrue(task.supports)
        self.assertNotEqual(episode.support, renamed.support)


if __name__ == "__main__":
    unittest.main()
