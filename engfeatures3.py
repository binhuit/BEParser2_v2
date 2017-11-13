from constant import PAD, ROOT


## Copyright 2010 Yoav Goldberg
##
## This file is part of easyfirst
##
##    easyfirst is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    easyfirst is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with easyfirst.  If not, see <http://www.gnu.org/licenses/>.


class BaselineFeatureExtractor:  # {{{
    LANG = 'ENG'

    def __init__(self):
        self.versions = None
        self.vocab = set()

    def extract(self, parsed, deps, i, sent=None):
        """
        i=T4:
           should I connect T4 and T5 in:
              t1 t2 t3 T4 T5 t6 t7 t8
           ?
           focus: f1=T4 f2=T5
           previous: p1=t3 p2=t2
           next:     n1=t6 n2=t7
        returns (feats1,feats2)
        where feats1 is a list of features for connecting T4->T5  (T4 is child)
        and   feats2 is a list of features for connecting T4<-T5  (T5 is child)
        """
        # LANG = self.LANG
        CC = ['CC', 'CONJ']
        IN = ['IN']

        j = i + 1
        features = []

        f1 = parsed[i]
        f2 = parsed[j]
        n1 = parsed[j + 1] if j + 1 < len(parsed) else PAD
        n2 = parsed[j + 2] if j + 2 < len(parsed) else PAD
        p1 = parsed[i - 1] if i - 1 > 0 else PAD
        p2 = parsed[i - 2] if i - 2 > 0 else PAD

        f1_form = f1['form']
        f2_form = f2['form']
        p1_form = p1['form']
        n1_form = n1['form']
        n2_form = n2['form']
        p2_form = p2['form']


        append = features.append
        append("pt1_%s_%s" % (p1_form, f1_form))
        append("t1t2_%s_%s" % (f1_form, f2_form))
        append("t2n_%s_%s" % (f2_form, n1_form))
        append("pn_%s_%s" % (p1_form, n1_form))


        return features

    # }}}


FeaturesExtractor = BaselineFeatureExtractor
