import os
import random
import copy
import isprojective
import time
from deps import DependenciesCollection
from engfeatures2 import FeaturesExtractor
from beam import Beam
from collections import defaultdict
from ml.ml import MultitronParameters, MulticlassModel
from ml_lib import MultitronParametersTest
from constant import ROOT



#########################################################################

class Oracle:
    """
    This class help check if the action on pending is valid on sentence
    """

    def __init__(self, sent):
        """
        Setup object in _init_sent function
        :param sent: sentence which use for check valid action
        """
        self._init_sent(sent)

    def _init_sent(self, sent):
        """
        setup for oracle object
        :param sent:
        :return:
        """
        # set sent attr of object
        self.sent = sent
        # create childs attr
        # self.childs[parent] is list of all (parent, child) pairs
        self.childs = defaultdict(set)
        for tok in sent:
            self.childs[tok['parent']].add((tok['parent'], tok['id']))

    def allow_connection(self, deps, parent, child):
        """

        :param deps: runtime deps
        :param parent: parent node
        :param child: child node
        :return: True if (parent, child) is valid. Otherwise
        """
        # (parent, child) is not in gold
        if child['parent'] != parent['id']:
            return False
        # There are exist at least one pair which child is a parent
        # and this arc is not in deps
        if len(self.childs[child['id']] - deps.deps) > 0:
            return False
        return True


class ParserEval(object):
    """
    This class help evaluate result of parser
    """

    def __init__(self):
        self.parses = []

    def add(self, parse):
        self.parses.append(parse)

    def eval(self):
        pass

    def output(self, output_file):
        pass


class TrainModel(object):
    """
    A model use in training phrase
    """

    def __init__(self, model_dir, beam_size):
        self.model_dir = model_dir
        self.feats_extractor = FeaturesExtractor()
        self.beam_size = beam_size
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        # A linear model classification with 2 class
        # 0: is left
        # 1: is right
        self.perceptron = MultitronParameters(2)

    def update(self, neg_state, pos_state):
        """
        rewarding features lead to correct action
        file features lead to wrong action
        :param neg_state:
        :param pos_state:
        :return:
        """
        # tick() increase variable store number of update
        # self.perceptron.tick()
        # features give correct action
        pos_actions = pos_state['actions']
        # right class
        for actions in pos_actions:
            # update paramater by plus one
            feats = actions['features']
            cls = actions['cls']
            self.perceptron.add(feats, cls, 1)
        # features give wrong action
        neg_actions = neg_state['actions']
        # wrong class
        for actions in neg_actions:
            feats = actions['features']
            cls = actions['cls']
            # update paramaters by minus one
            self.perceptron.add(feats, cls, -1)

    def save(self, iter):
        # save model paramaters
        weight_file = 'weight.%s' % iter
        weight_file_path = os.path.join(self.model_dir, weight_file)
        self.perceptron.dump_fin(file(weight_file_path, 'w'))

    def tick(self):
        self.perceptron.tick()

    def featex(self, pending, deps, i):
        # called by parser object
        return self.feats_extractor.extract(pending, deps, i)

    def get_score(self, features):
        # return a dict of score in aspact of class
        return self.perceptron.get_scores(features)


class TestModel(object):
    """
    Model use to test
    """

    def __init__(self, model_dir, beam_size, iter='FINAL'):
        self.feats_extractor = FeaturesExtractor()
        weight_name = 'weight.' + iter
        weight_path = os.path.join(model_dir, weight_name)
        # load already trained model
        self.perceptron = MulticlassModel(weight_path)
        self.beam_size = beam_size

    def featex(self, pending, deps, i):
        return self.feats_extractor.extract(pending, deps, i)

    def get_score(self, features):
        return self.perceptron.get_scores(features)


def get_state(pending, actions=None, score=0.0, deps=DependenciesCollection(),
              fcached=None, scached=None, valid=True):
        deps = copy.copy(deps)
        if fcached is None:
            fcached = {}
        if scached is None:
            scached = {}
        if actions is None:
            actions = []
        return {
            'pending': pending,
            'actions': actions,
            'score': score,
            'deps': deps,
            'valid': valid,
            'fcached': fcached,
            'scached': scached

        }


def build_gold(sent):
    deps = DependenciesCollection()
    for token in sent[1:]:
        child = token
        parent = sent[child['parent']]
        deps.add(parent, child)
    return deps


class Parser(object):
    """
    A heart of program, train and test
    """

    def __init__(self, model):
        # model of parser
        self.model = model

    def parse(self, sent):
        sent = [ROOT] + sent
        init_state = get_state(sent)
        beam = Beam(self.model.beam_size)
        beam.add(init_state)
        for step in range(len(sent) - 1):
            next_beam = Beam(self.model.beam_size)
            for state in beam:
                pending = state['pending']
                prev_score = state['score']
                prev_actions = ['actions']
                scached, fcached = self._generate_cached(state)
                for i, (tok1, tok2) in enumerate(zip(pending, pending[1:])):
                    cached_idx = (tok1['id'], tok2['id'])
                    scores = scached[cached_idx]
                    lc_feats = fcached[cached_idx]
                    for clas, score in enumerate(scores):
                        arc = self._get_action(clas, tok1, tok2)
                        next_action = self._wrap_action(clas, lc_feats, arc)
                        actions = prev_actions + [next_action]
                        n_score = prev_score + score

                        if next_beam.is_empty() or n_score > next_beam.min_score():
                            new_state = self._gen_state(i, pending, actions,n_score,
                                                        scached, fcached, arc, state)
                            next_beam.add(new_state)
            beam = next_beam
        deps = beam.top()['deps']
        return deps

    def train(self, sent):
        def check_equal(deps1, deps2):
            gold_arcs = deps1.deps
            beam_arcs = deps2.deps
            if not gold_arcs.difference(beam_arcs):
                return True
            else:
                return False

        self.model.tick()
        sent = [ROOT] + sent
        oracle = Oracle(sent)
        gold_deps = build_gold(sent)
        init_state = get_state(sent)
        beam = Beam(self.model.beam_size)
        beam.add(init_state)
        valid_action = Beam(beam_size=1)
        for step in range(len(sent) - 1):
            # beam, valid_action = self._extend_beam(beam, oracle)
            next_beam = Beam(self.model.beam_size)
            valid_action.clear()
            for state in beam:
                pending = state['pending']
                prev_score = state['score']
                prev_actions = state['actions']
                deps = state['deps']
                stt = state['valid']
                scached, fcached = self._generate_cached(state)
                for i, (tok1, tok2) in enumerate(zip(pending, pending[1:])):
                    cached_idx = (tok1['id'], tok2['id'])
                    scores = scached[cached_idx]
                    lc_feats = fcached[cached_idx]

                    for clas, score in scores.iteritems():
                        arc = self._get_action(clas, tok1, tok2)
                        actions = prev_actions + [self._wrap_action(clas, lc_feats, arc)]
                        if stt:
                            is_valid = self._check_valid(arc, deps, oracle)
                        else:
                            is_valid = False
                        n_score = prev_score + score
                        if next_beam.is_empty() or n_score > next_beam.min_score() or is_valid:
                            new_state = self._gen_state(i, pending, actions, n_score, scached,
                                                        fcached, arc, state, is_valid)
                            if next_beam.is_empty() or n_score > next_beam.min_score():
                                next_beam.add(new_state)
                            if is_valid:
                                valid_action.add(new_state)
            beam = next_beam
            if not beam.has_element(valid_action.top()):
                beam_top = beam.top()
                self.model.update(beam_top, valid_action.top())
                break
        else:
            best_state = beam.top()
            predicted_deps = best_state['deps']
            if not check_equal(gold_deps, predicted_deps):
                self.model.update(best_state, valid_action.top())

    def save_weight(self, iter):
        # save model
        self.model.save(iter)


    def _apply_action(self, arc, state):
        # return new pending and new deps
        deps = copy.deepcopy(state['deps'])
        pending = state['pending']
        # unpacking arc
        child, parent = arc
        # add arc to deps
        deps.add(parent, child)
        # remove child
        child_idx = pending.index(child)
        if child_idx == 0:
            new_pending = pending[1:]
        elif child_idx == len(pending) - 1:
            new_pending = pending[:-1]
        else:
            new_pending = pending[:child_idx] + pending[child_idx+1:]
        return new_pending, deps

    def _check_valid(self, arc, deps, oracle):
        # use oracle to check valid status of an action
        return oracle.allow_connection(deps, arc[1], arc[0])

    def _get_action(self, clas, tok1, tok2):
        """

        :param clas:
        :param tok1:
        :param tok2:
        :return: children, parent
        """
        if clas == 0:
            return tok1, tok2
        else:
            return tok2, tok1

    def _get_cached(self, pos, cached, pending):
        n_cached = copy.copy(cached)
        frm = pos - 4
        to = pos + 4
        if frm < 0: frm = 0
        if to >= len(pending): to = len(pending) - 1
        for i in range(frm, to):
            try:
                del n_cached[(pending[i]['id'], pending[i + 1]['id'])]
            except KeyError:
                pass
        return n_cached

    def _wrap_action(self, cls, features, arc):
        # arc_string = "%s --> %s" % (arc.parent['form'], arc.child['form'])
        return {'cls': cls,
                'features': features}

    def _generate_cached(self, state):
        deps = state['deps']
        pending = state['pending']
        scached = state['scached']
        fcached = state['fcached']
        for i, (tok1, tok2) in enumerate(zip(pending, pending[1:])):
            cached_idx = (tok1['id'], tok2['id'])
            if not cached_idx in fcached:
                features = self.model.featex(pending, deps, i)
                fcached[cached_idx] = features
                scached[cached_idx] = self.model.get_score(features)
        return scached, fcached

    def _gen_state(self, i, pending, actions, score, scached, fcached, arc, state, valid=True):
        n_scached = self._get_cached(i, scached, pending)
        n_fcached = self._get_cached(i, fcached, pending)
        n_pending, n_deps = self._apply_action(arc, state)
        new_state = get_state(n_pending, actions, score,
                              n_deps, n_fcached, n_scached, valid)
        return new_state


def read_corpus(filename):
    """
    Reading corpus file and yield sentence
    :param filename: corpus file name
    :return: list of tokens
    """
    dependency_corpus = open(filename)
    sent = []
    try:
        for line in dependency_corpus:
            line = line.strip().split()
            if line:
                sent.append(line_to_tok(line))
            elif sent:
                yield sent
                sent = []
    finally:
        if sent:
            yield sent
        dependency_corpus.close()


def line_to_tok(line):
    return {
        'id': int(line[0]),
        'form': line[1],
        'tag': line[4],
        'parent': int(line[6]),
        'prel': line[7]
    }


def train(model_dir, train_data, iter, beam_size):
    global update_time
    global before_train
    model = TrainModel(model_dir, beam_size)
    parser = Parser(model)
    total_time = 0.0
    for i in range(1, iter + 1):
        print 'iter %d' % i
        random.shuffle(train_data)
        start = time.time()
        for num, sent in enumerate(train_data):
            parser.train(sent)
            if num % 100 == 0:
                print num
        iter_time = time.time() - start
        total_time += iter_time
        print 'Time train: %s' % iter_time
        if i % 1 == 0:
            parser.save_weight(str(i))
    print "Total training time: %s" % total_time
    parser.save_weight('FINAL')


def count_correct(parsed, gold):
    parsed_arcs = parsed.deps
    gold_arcs = gold.deps
    return len(parsed_arcs.intersection(gold_arcs))


def test(model_dir, test_data, output_file, beam_size, iter='FINAL'):
    model = TestModel(model_dir, beam_size, iter)
    parser = Parser(model)
    correct = 0.0
    total = 0.0
    data = []
    for i, sent in enumerate(test_data):
        dependency_tree = parser.parse(sent)
        gold_tree = build_gold([ROOT] + sent)
        num_correct = count_correct(dependency_tree, gold_tree)
        correct += num_correct
        total += len(sent)
        if (len(sent) - num_correct > 0):
            data.append("cau %d: %d/%d" % (i, num_correct, len(sent)))
            data.append(" ".join(token['form'] for token in sent))
    with open("wrong_predict_sent_beam_4", "w") as f:
        for line in data:
            f.write(line + "\n")

    print 'Correct: %d' % correct
    print 'Total: %d' % total
    print "Accuracy: " + str(correct / total)


def main():
    model_dir = 'test_model'
    train_file = 'data'
    test_file = 'data'
    beam_size = 1
    output_file = None
    iter = 10
    is_train = False
    if is_train:
        train_data = list(read_corpus(train_file))
        print len(train_data)
        train_sents = [s for s in train_data if isprojective.is_projective(s)]
        print len(train_sents)
        train(model_dir, train_sents, iter, beam_size)
    else:
        test_data = list(read_corpus(test_file))
        test(model_dir, test_data, output_file, beam_size)


main()
