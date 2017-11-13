import os
import random
from deps import DependenciesCollection
from engfeatures3 import FeaturesExtractor
from beam import Beam
from collections import defaultdict, namedtuple
from ml.ml import MultitronParameters, MulticlassModel
from ml_lib import MultitronParametersTest
from constant import ROOT, PAD
import copy
import isprojective



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
        # model_dir to save model
        self.model_dir = model_dir
        # object use to extract feature in pending
        self.feats_extractor = FeaturesExtractor()
        self.beam_size = beam_size
        # if model_dir is not exist, create it
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
    def __init__(self, model_dir, beam_size, iter = 'FINAL'):
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

class Parser(object):
    """
    A heart of program, train and test
    """
    def __init__(self, model):
        # model of parser
        self.model = model

    def parse(self, sent):
        # parse one sent according to current model paramaters
        # ROOT token at begining of pending
        sent = [ROOT] + sent
        # start state
        init_state = self._get_state(sent)
        # create a beam
        beam = Beam(self.model.beam_size)
        # add state to beam
        beam.add(init_state)
        # loop until only one tree left
        for step in range(len(sent) - 1):
            # beam of next step
            beam = self._extend_beam_for_test(beam)
        # result of parse
        deps = beam.top()['deps']
        return deps

    def train(self, sent):
        # update paramaters with one sent
        # ROOT token at begining of pending
        sent = [ROOT] + sent
        # oracle object to check valid action
        oracle = Oracle(sent)
        # gold_deps for full update
        gold_deps = self._build_gold(sent)
        # create start state
        init_state = self._get_state(sent)
        # create beam
        beam = Beam(self.model.beam_size)
        # add state to beam
        beam.add(init_state)
        # correct action with highest score at one step
        valid_action = None
        for step in range(len(sent) - 1):
            beam, valid_action = self._extend_beam(beam, oracle)
            # if beam not contain valid action in it, update
            if not beam.has_element(valid_action):
                beam_top = beam.top()
                self.model.update(beam_top, valid_action)
                break
        else:
            beam_top = beam.top()
            beam_deps = beam_top['deps']
            # if final deps is not like gold_deps, do full update
            if not self._check_equal(gold_deps, beam_deps):
                self.model.update(beam_top, valid_action)

    def _check_equal(self, gold_deps, beam_deps):
        "check if two set is equal"
        gold_arcs = gold_deps.deps
        beam_arcs = beam_deps.deps
        # if two set joint and return none, they are identical
        if not gold_arcs.difference(beam_arcs):
            return True
        else:
            return False

    def save_weight(self, iter):
        # save model
        self.model.save(iter)

    def _build_gold(self, sent):
        # build gold deps
        deps = DependenciesCollection()
        for token in sent[1:]:
            child = token
            parent = sent[child['parent']]
            deps.add(parent, child)
        return deps

    def _get_state(self, pending, actions=[], score=float('-inf'), clas=None,
                   deps=DependenciesCollection(), valid=True, fcached={}, scached={}):
        """
        state in beam
        :param pending: list of token
        :param features: global features until prv action
        :param score: score of this state
        :param clas: class of prev action
        :param deps: current deps
        :param valid: is this state valid
        :return: a dict
        """
        # copy pending
        pending = list(pending)
        # copy features
        features = copy.copy(actions)
        # copy deps
        deps = copy.copy(deps)
        return {
            'pending': pending,
            'actions': actions,
            'score': score,
            'deps': deps,
            'valid': valid,
            'fcached': fcached,
            'scached': scached

        }

    def _apply_action(self, arc, state):
        # return new pending and new deps
        deps = copy.deepcopy(state['deps'])
        pending = list(state['pending'])
        # unpacking arc
        child, parent = arc
        # add arc to deps
        deps.add(parent, child)
        # remove child
        pending.remove(child)
        return pending, deps

    def _check_valid(self, arc, deps, oracle):
        # use oracle to check valid status of an action
        return oracle.allow_connection(deps, arc.parent, arc.child)

    def _extract_state(self, state):
        # unpack state
        return state['pending'], state['score'], state['actions'], state['deps']\
            ,state['valid']

    def _get_action(self, clas, tok1, tok2):
        """
        
        :param clas: 
        :param tok1: 
        :param tok2: 
        :return: children, parent
        """
        arc = namedtuple('arc',['child', 'parent'])
        if clas == 0:
            return arc(tok1, tok2)
        else:
            return arc(tok2, tok1)

    def _get_cached(self, pos, cached, pending):
        n_cached = copy.copy(cached)
        frm = pos - 4
        to = pos + 4
        if frm < 0: frm = 0
        if to >= len(pending): to = len(pending) - 1
        for i in range(frm, to):
            try:
                del n_cached[(pending[i]['id'], pending[i+1]['id'])]
            except KeyError:
                pass
        return n_cached

    def _wrap_action(self, cls, features, arc):
        arc_string = "%s --> %s" % (arc.parent['form'], arc.child['form'])
        return {'cls': cls,
                'features': features,
                'desc': arc_string}

    def _extend_beam_for_test(self, beam):
        # return beam for next step
        new_beam = Beam(self.model.beam_size)
        # go over all state in beam
        for state in beam:
            # unpacking state
            pending, prev_score, prev_actions, deps, _ = self._extract_state(state)
            scached = copy.copy(state['scached'])
            fcached = copy.copy(state['fcached'])
            # scached = state['scached']
            # fcached = state['fcached']
            for i, (tok1, tok2) in enumerate(zip(pending, pending[1:])):
                cached_idx = (tok1['id'], tok2['id'])
                if cached_idx in fcached:
                    lc_feats = fcached[cached_idx]
                else:
                    lc_feats = self.model.featex(pending, deps, i)
                    fcached[cached_idx] = lc_feats
                # score of local features
                if cached_idx in scached:
                    scores = scached[cached_idx]
                else:
                    scores = self.model.get_score(lc_feats)
                    scached[cached_idx] = scores

                for clas, score in enumerate(scores):
                    arc = self._get_action(clas, tok1, tok2)
                    actions = prev_actions + [self._wrap_action(i, lc_feats, arc)]
                    n_scached = self._get_cached(i, scached, pending)
                    n_fcached = self._get_cached(i, fcached, pending)
                    n_pending, n_deps = self._apply_action(arc, state)
                    # try starter score is zero
                    if prev_score == float('-inf'):
                        n_score = score
                    else:
                        n_score = prev_score + score
                    new_state = self._get_state(n_pending, actions, n_score,
                                                clas, n_deps, n_fcached, n_scached)

                    new_beam.add(new_state)
        return new_beam



    def _extend_beam(self, beam, oracle):
        new_beam = Beam(self.model.beam_size)
        valid_action = Beam(beam_size=1)
        for state in beam:
            pending, prev_score, prev_actions, deps, stt = self._extract_state(state)
            scached = copy.copy(state['scached'])
            fcached = copy.copy(state['fcached'])
            # scached = state['scached']
            # fcached = state['fcached']
            for i, (tok1, tok2) in enumerate(zip(pending, pending[1:])):
                cached_idx = (tok1['id'], tok2['id'])
                if cached_idx in fcached:
                    lc_feats = fcached[cached_idx]
                else:
                    lc_feats = self.model.featex(pending, deps, i)
                    fcached[cached_idx] = lc_feats
                # score of local features
                if cached_idx in scached:
                    scores = scached[cached_idx]
                else:
                    scores = self.model.get_score(lc_feats)
                    scached[cached_idx] = scores

                for clas, score in scores.iteritems():
                    arc = self._get_action(clas, tok1, tok2)
                    actions = prev_actions + [self._wrap_action(clas, lc_feats, arc)]
                    # stt ensure all action before in state is valid
                    if stt:
                        is_valid = self._check_valid(arc, deps, oracle)
                    else:
                        is_valid = False
                    n_scached = self._get_cached(i, scached, pending)
                    n_fcached = self._get_cached(i, fcached, pending)
                    n_pending, n_deps = self._apply_action(arc, state)
                    if prev_score == float('-inf'):
                        n_score = score
                    else:
                        n_score = prev_score + score
                    new_state = self._get_state(n_pending, actions, n_score,
                                                clas, n_deps, is_valid, n_fcached, n_scached)
                    if new_beam.is_empty() or n_score > new_beam.min_score():
                        new_beam.add(new_state)
                    if is_valid:
                        valid_action.add(new_state)
        return new_beam, valid_action.top()


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
    model = TrainModel(model_dir, beam_size)
    parser = Parser(model)
    for i in range(1, iter + 1):
        model.tick()
        print 'iter %d' % i
        random.shuffle(train_data)
        for num, sent in enumerate(train_data):
            parser.train(sent)
            if num % 100 == 0:
                print num
        if i % 1 == 0:
            parser.save_weight(str(i))
    parser.save_weight('FINAL')

def count_correct(parsed, gold):
    parsed_arcs = parsed.deps
    gold_arcs = gold.deps
    return len(parsed_arcs.intersection(gold_arcs))

def test(model_dir, test_data, output_file, beam_size, iter = 'FINAL'):
    model = TestModel(model_dir, beam_size, iter)
    parser = Parser(model)
    correct = 0.0
    total = 0.0
    data = []
    for i, sent in enumerate(test_data):
        dependency_tree = parser.parse(sent)
        gold_tree = parser._build_gold([ROOT] + sent)
        num_correct = count_correct(dependency_tree, gold_tree)
        correct += num_correct
        total += len(sent)
        if (len(sent) - num_correct > 0):
            data.append("cau %d: %d/%d" % (i, num_correct, len(sent)))
            data.append(" ".join(token['form'] for token in sent))
    with open("wrong_predict_sent_beam_4","w") as f:
        for line in data:
            f.write(line+"\n")

    print 'Correct: %d' % correct
    print 'Total: %d' % total
    print "Accuracy: " + str(correct/total)

def main():
    model_dir = 'test_model_beam_4'
    train_file = 'data'
    test_file = 'data'
    beam_size = 4
    output_file = None
    iter = 30
    is_train = True
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
