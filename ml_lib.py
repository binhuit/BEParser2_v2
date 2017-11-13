class MulticlassParamData:
   def __init__(self, nclasses):
      self.lastUpd = {}
      self.acc = {}
      self.w = {}
      for i in range(nclasses):
         self.lastUpd[i]=0.0
         self.acc[i]=0.0
         self.w[i]=0.0


class MultitronParametersTest:

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.now = 0
        self.W = {}

    def tick(self): self.now += 1

    def get_scores(self, features):
        scores = {}
        for i in xrange(self.nclasses):
            scores[i] = 0.0
        for f in features:
            try:
                p = self.W[f]
                for c in xrange(self.nclasses):
                    scores[c] += p.w[c]
            except KeyError:
                pass
        return scores

    def add(self, features, clas, amount):
        for f in features:
            try:
                p = self.W[f]
            except KeyError:
                p = MulticlassParamData(self.nclasses)
                self.W[f] = p

            p.acc[clas] += (self.now - p.lastUpd[clas]) * p.w[clas]
            p.w[clas] += amount
            p.lastUpd[clas] = self.now

    def dump_fin(self, out):
        # write the average
        for f in self.W.keys():
            out.write("%s" % f)
            for c in xrange(self.nclasses):
                p = self.W[f]
                out.write(" %s " % ((p.acc[c] + ((self.now - p.lastUpd[c]) * p.w[c])) / self.now))
                # out.write(" %s " % (p.w[c]))
            out.write("\n")

