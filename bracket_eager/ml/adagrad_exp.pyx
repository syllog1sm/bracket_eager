
cdef class SparseAveragedAdagradParameters: #{{{
   """
   AdaGrad with perceptron loss and averagin.
   A quick-hack and inefficient implementation that will re-use SparseMulticlassParamData.
   The downside is double the memory usage for no reason.
   """
   cdef:
      int nclasses
      int now
      double rate
      dict W
      dict G  # these are the gradient^2 sums

      double* scores # (re)used in calculating prediction
   
   def __cinit__(self, nclasses):
      self.scores = <double *>malloc(nclasses*sizeof(double))

   cpdef getW(self, clas): 
      d={}
      cdef SparseMulticlassParamData p
      for f,p in self.W.iteritems():
         d[f] = p.w_for_clas(clas)
      return d

   def __init__(self, nclasses, double rate=1.0):
      self.nclasses = nclasses
      self.W = {}
      self.rate = rate
      self.now = 0
      #self.G = {}

   cdef _tick(self):
      self.now=self.now+1

   def tick(self): 
      self._tick()

   cpdef update_me(self, list features, list good_classes):
      self._get_normed_exp_scores(features, self.scores)
      cdef double gnormalizer = 0
      cdef int c
      cdef double s

      for c in good_classes:
         gnormalizer += self.scores[c]

      for c in good_classes:
         if gnormalizer == 0:
            #print "adding0",c,0
            self.add(features, c, 0) # add 0 instead of skipping to initialize the data structures if they are not initialized.
            continue # TODO wtf?
         else:
            #print "adding",c,scores[c]/gnormalizer
            self.add(features, c, self.scores[c]/gnormalizer)
      for c in xrange(self.nclasses):
         if self.scores[c] > 0:
         #print "adding",c,-scores[c]/normalizer
            self.add(features, c, -self.scores[c])

   cpdef add(self, list features, int clas, double amount):
      cdef SparseMulticlassParamData p
      #cdef SparseMulticlassParamData g
      cdef double to_add
      cdef double delta = 0.1
      cdef double h
      for f in features:
         try:
            p = self.W[f]
            #g = self.G[f]
         except KeyError:
            p = SparseMulticlassParamData()
            #g = SparseMulticlassParamData()
            self.W[f] = p
            #self.G[f] = g
         # calculate the addition:
         p.add_to_clas_acc2(clas, (1*amount)*(1*amount))
         h = sqrt(p.acc2_for_clas(clas)) + delta
         to_add = (1*amount) * (self.rate / h)
         p.add_to_clas(clas, to_add, self.now)

         # all feature weights are 1, and 1^2 = 1

   cpdef add_rda(self, list features, int clas, double amount):
      """
      RDA update based on:
         http://code.google.com/p/factorie/source/browse/src/main/scala/cc/factorie/optimize/AdaGradRDA.scala
      l1 is not working.
      """
      self.rate = 10
      cdef double l1 = 0.01
      cdef SparseMulticlassParamData p
      #cdef SparseMulticlassParamData g
      cdef double to_add
      cdef double delta = 0.1
      cdef double h
      cdef double acc
      for f in features:
         try:
            p = self.W[f]
            #g = self.G[f]
         except KeyError:
            p = SparseMulticlassParamData()
            #g = SparseMulticlassParamData()
            self.W[f] = p
            #self.G[f] = g
         # calculate the addition:
         p.add_to_clas_acc2(clas, (1*amount)*(1*amount))
         p.add_to_clas_acc(clas, 1*amount)
         h = sqrt(p.acc2_for_clas(clas)) + delta
         to_add = 1.0 / ((1.0 / self.rate) * h)
         acc = p.acc_for_clas(clas)
         #print "acc",acc,l1*self.now
         if acc >= 0:
            acc -= (l1 * self.now)
            if acc > 0:
               #print "up1:",to_add*acc
               p.set_clas_w_to(clas, to_add * acc, self.now, 0)
         else:
            acc += (l1 * self.now)
            if acc < 0:
               #print "up2:",to_add*acc
               p.set_clas_w_to(clas, to_add * acc, self.now, 0)
         # all feature weights are 1, and 1^2 = 1

   cpdef get_scores(self, features):
      cdef SparseMulticlassParamData p
      cdef int i
      cdef double w
      for i in xrange(self.nclasses):
         self.scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            p.add_w_to_scores(self.scores)
         except KeyError: pass
      cdef double tot = 0
      res={}
      for i in xrange(self.nclasses):
         res[i] = self.scores[i]
      return res

   cdef _get_normed_exp_scores(self, features, double* scores):
      """
      see: http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
      for explanation about the "a" subtraction.
      """
      cdef SparseMulticlassParamData p
      cdef int i
      cdef double w
      cdef double e
      cdef double a
      for i in xrange(self.nclasses):
         scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            p.add_w_to_scores(scores)
         except KeyError: pass
      cdef double tot = 0
      a = self.scores[0]
      for i in xrange(self.nclasses):
         if scores[i] > a:
            a = scores[i]
         
      for i in xrange(self.nclasses):
         e = exp(scores[i] - a)
         scores[i] = e
         tot += e
      for i in xrange(self.nclasses):
         scores[i] = scores[i] / tot

   cpdef get_normed_exp_scores(self, features):
      self._get_normed_exp_scores(features, self.scores)
      res={}
      for i in xrange(self.nclasses):
         res[i] = self.scores[i]
      return res

      #cdef SparseMulticlassParamData p
      #cdef int i
      #cdef double w
      #cdef double e
      #cdef double a
      #for i in xrange(self.nclasses):
      #   self.scores[i]=0
      #for f in features:
      #   try:
      #      p = self.W[f]
      #      p.add_w_to_scores(self.scores)
      #   except KeyError: pass
      #cdef double tot = 0
      #res={}
      #a = self.scores[0]
      #for i in xrange(self.nclasses):
      #   if self.scores[i] > a:
      #      a = self.scores[i]
      #   
      #for i in xrange(self.nclasses):
      #   e = exp(self.scores[i] - a)
      #   self.scores[i] = e
      #   tot += e
      #for i in xrange(self.nclasses):
      #   res[i] = self.scores[i] / tot
      #   #print "s,e(s):",i,self.scores[i],math.exp(self.scores[i])
      #   #res[i] = math.exp(self.scores[i]) / tot
      #return res

   cpdef get_best_class(self, features):
      cdef SparseMulticlassParamData p
      cdef int i
      cdef double w
      cdef int best_i
      cdef double best_score
      for i in xrange(self.nclasses):
         self.scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            p.add_w_to_scores(self.scores)
         except KeyError: pass
      cdef double tot = 0
      best_score = self.scores[0]
      best_i = 0
      for i in xrange(self.nclasses):
         if self.scores[i] > best_score:
            best_i = i
            best_score = self.scores[i]
      return (best_score,best_i)

   def finalize(self):
      cdef SparseMulticlassParamData p
      # average
      for f in self.W.keys():
         p = self.W[f]
         p.finalize(self.now)

   def dump(self, out=sys.stdout, sparse=False):
      cdef SparseMulticlassParamData p
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         out.write("%s" % f)
         for c in xrange(self.nclasses):
            p = self.W[f]
            w = p.w_for_clas(c)
            if sparse:
               if w != 0:
                  out.write(" %s:%s" % (c,w))
            else:
               out.write(" %s" % w)
         out.write("\n")

   def dump_fin(self,out=sys.stdout, sparse=False):
      cdef SparseMulticlassParamData p
      # write the average
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         out.write("%s" % f)
         for c in xrange(self.nclasses):
            p = self.W[f]
            w = p.avgd_w_for_clas(c, self.now)
            if sparse:
               if w != 0:
                  out.write(" %s:%s" % (c,w))
            else:
               out.write(" %s " % (w))
         out.write("\n")

#}}}

cdef class SparseRegularizedAdagradParameters: #{{{
   """
   """
   cdef:
      int nclasses
      int now
      double rate
      dict W
      dict G  # these are the gradient^2 sums
      dict last_regs

      double* scores # (re)used in calculating prediction
   
   def __cinit__(self, nclasses):
      self.scores = <double *>malloc(nclasses*sizeof(double))

   cpdef getW(self, clas): 
      d={}
      cdef SparseMulticlassParamData p
      for f,p in self.W.iteritems():
         d[f] = p.w_for_clas(clas)
      return d

   def __init__(self, nclasses, double rate=1.0):
      self.nclasses = nclasses
      self.W = {}
      self.rate = rate
      self.now = 0
      #self.G = {}
      self.last_regs = {}

   cdef _tick(self):
      self.now=self.now+1

   def tick(self): 
      self._tick()

   cdef regularize(self, features):
      cdef SparseMulticlassParamData p
      cdef int num
      cdef double delta = 0.1
      cdef double l1 = 0.01/1000
      cdef int c
      cdef double shrink
      cdef double h
      cdef double w
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            continue
         last_reg = self.last_regs.get(f, 0)
         num = self.now - last_reg
         for c in xrange(self.nclasses):
            h = sqrt(p.acc2_for_clas(c)) + delta
            shrink = num * l1 * (self.rate / h)
            w = p.w_for_clas(c)
            if w > shrink:
               w = w - shrink
            elif w < -shrink:
               w = w + shrink
            else:
               w = 0
            p.set_clas_w_to(c, w, self.now, 0)
         self.last_regs[f] = self.now

   cpdef update_me(self, list features, list good_classes):
      self._get_normed_exp_scores(features, self.scores)
      cdef double gnormalizer = 0
      cdef int c
      cdef double s

      for c in good_classes:
         gnormalizer += self.scores[c]

      for c in good_classes:
         if gnormalizer == 0:
            #print "adding0",c,0
            self.add(features, c, 0) # add 0 instead of skipping to initialize the data structures if they are not initialized.
            continue # TODO wtf?
         else:
            #print "adding",c,scores[c]/gnormalizer
            self.add(features, c, self.scores[c]/gnormalizer)
      for c in xrange(self.nclasses):
         if self.scores[c] > 0:
         #print "adding",c,-scores[c]/normalizer
            self.add(features, c, -self.scores[c])

   cpdef add(self, list features, int clas, double amount):
      cdef SparseMulticlassParamData p
      #cdef SparseMulticlassParamData g
      cdef double to_add
      cdef double delta = 0.1
      cdef double h
      for f in features:
         try:
            p = self.W[f]
            #g = self.G[f]
         except KeyError:
            p = SparseMulticlassParamData()
            #g = SparseMulticlassParamData()
            self.W[f] = p
            #self.G[f] = g
         # calculate the addition:
         p.add_to_clas_acc2(clas, (1*amount)*(1*amount))
         h = sqrt(p.acc2_for_clas(clas)) + delta
         to_add = (1*amount) * (self.rate / h)
         p.add_to_clas(clas, to_add, self.now)

         # all feature weights are 1, and 1^2 = 1

   cpdef add_rda(self, list features, int clas, double amount):
      """
      RDA update based on:
         http://code.google.com/p/factorie/source/browse/src/main/scala/cc/factorie/optimize/AdaGradRDA.scala
      l1 is not working.
      """
      self.rate = 10
      cdef double l1 = 0.01
      cdef SparseMulticlassParamData p
      #cdef SparseMulticlassParamData g
      cdef double to_add
      cdef double delta = 0.1
      cdef double h
      cdef double acc
      for f in features:
         try:
            p = self.W[f]
            #g = self.G[f]
         except KeyError:
            p = SparseMulticlassParamData()
            #g = SparseMulticlassParamData()
            self.W[f] = p
            #self.G[f] = g
         # calculate the addition:
         p.add_to_clas_acc2(clas, (1*amount)*(1*amount))
         p.add_to_clas_acc(clas, 1*amount)
         h = sqrt(p.acc2_for_clas(clas)) + delta
         to_add = 1.0 / ((1.0 / self.rate) * h)
         acc = p.acc_for_clas(clas)
         #print "acc",acc,l1*self.now
         if acc >= 0:
            acc -= (l1 * self.now)
            if acc > 0:
               #print "up1:",to_add*acc
               p.set_clas_w_to(clas, to_add * acc, self.now, 0)
         else:
            acc += (l1 * self.now)
            if acc < 0:
               #print "up2:",to_add*acc
               p.set_clas_w_to(clas, to_add * acc, self.now, 0)
         # all feature weights are 1, and 1^2 = 1

   cpdef get_scores(self, features):
      self.regularize(features)
      cdef SparseMulticlassParamData p
      cdef int i
      cdef double w
      for i in xrange(self.nclasses):
         self.scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            p.add_w_to_scores(self.scores)
         except KeyError: pass
      cdef double tot = 0
      res={}
      for i in xrange(self.nclasses):
         res[i] = self.scores[i]
      return res

   cdef _get_normed_exp_scores(self, features, double* scores):
      """
      see: http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
      for explanation about the "a" subtraction.
      """
      self.regularize(features)
      cdef SparseMulticlassParamData p
      cdef int i
      cdef double w
      cdef double e
      cdef double a
      for i in xrange(self.nclasses):
         scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            p.add_w_to_scores(scores)
         except KeyError: pass
      cdef double tot = 0
      a = self.scores[0]
      for i in xrange(self.nclasses):
         if scores[i] > a:
            a = scores[i]
         
      for i in xrange(self.nclasses):
         e = exp(scores[i] - a)
         scores[i] = e
         tot += e
      for i in xrange(self.nclasses):
         scores[i] = scores[i] / tot

   cpdef get_normed_exp_scores(self, features):
      self._get_normed_exp_scores(features, self.scores)
      res={}
      for i in xrange(self.nclasses):
         res[i] = self.scores[i]
      return res

      #cdef SparseMulticlassParamData p
      #cdef int i
      #cdef double w
      #cdef double e
      #cdef double a
      #for i in xrange(self.nclasses):
      #   self.scores[i]=0
      #for f in features:
      #   try:
      #      p = self.W[f]
      #      p.add_w_to_scores(self.scores)
      #   except KeyError: pass
      #cdef double tot = 0
      #res={}
      #a = self.scores[0]
      #for i in xrange(self.nclasses):
      #   if self.scores[i] > a:
      #      a = self.scores[i]
      #   
      #for i in xrange(self.nclasses):
      #   e = exp(self.scores[i] - a)
      #   self.scores[i] = e
      #   tot += e
      #for i in xrange(self.nclasses):
      #   res[i] = self.scores[i] / tot
      #   #print "s,e(s):",i,self.scores[i],math.exp(self.scores[i])
      #   #res[i] = math.exp(self.scores[i]) / tot
      #return res

   cpdef get_best_class(self, features):
      cdef SparseMulticlassParamData p
      cdef int i
      cdef double w
      cdef int best_i
      cdef double best_score
      for i in xrange(self.nclasses):
         self.scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            p.add_w_to_scores(self.scores)
         except KeyError: pass
      cdef double tot = 0
      best_score = self.scores[0]
      best_i = 0
      for i in xrange(self.nclasses):
         if self.scores[i] > best_score:
            best_i = i
            best_score = self.scores[i]
      return (best_score,best_i)

   def finalize(self):
      cdef SparseMulticlassParamData p
      # average
      for f in self.W.keys():
         p = self.W[f]
         p.finalize(self.now)

   def dump(self, out=sys.stdout, sparse=False):
      cdef SparseMulticlassParamData p
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         out.write("%s" % f)
         for c in xrange(self.nclasses):
            p = self.W[f]
            w = p.w_for_clas(c)
            if sparse:
               if w != 0:
                  out.write(" %s:%s" % (c,w))
            else:
               out.write(" %s" % w)
         out.write("\n")

   def dump_fin(self,out=sys.stdout, sparse=False):
      cdef SparseMulticlassParamData p
      # write the average
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         out.write("%s" % f)
         for c in xrange(self.nclasses):
            p = self.W[f]
            w = p.avgd_w_for_clas(c, self.now)
            if sparse:
               if w != 0:
                  out.write(" %s:%s" % (c,w))
            else:
               out.write(" %s " % (w))
         out.write("\n")

#}}}

cdef class SparseRDAAdagradParameters: #{{{
   """
   AdaGrad with perceptron loss and averagin.
   A quick-hack and inefficient implementation that will re-use SparseMulticlassParamData.
   The downside is double the memory usage for no reason.
   """
   cdef:
      int nclasses
      int now
      double rate
      dict W
      dict G  # these are the gradient^2 sums

      double* scores # (re)used in calculating prediction
   
   def __cinit__(self, nclasses):
      self.scores = <double *>malloc(nclasses*sizeof(double))

   cpdef getW(self, clas): 
      d={}
      cdef SparseMulticlassParamData p
      for f,p in self.W.iteritems():
         d[f] = p.w_for_clas(clas)
      return d

   def __init__(self, nclasses, double rate=1.0):
      self.nclasses = nclasses
      self.W = {}
      self.rate = rate
      self.now = 0
      #self.G = {}

   cdef _tick(self):
      self.now=self.now+1

   def tick(self): 
      self._tick()

   cpdef update_me(self, list features, list good_classes):
      self._get_normed_exp_scores(features, self.scores)
      cdef double gnormalizer = 0
      cdef int c
      cdef double s

      for c in good_classes:
         gnormalizer += self.scores[c]

      for c in good_classes:
         if gnormalizer == 0:
            #print "adding0",c,0
            self.add(features, c, 0) # add 0 instead of skipping to initialize the data structures if they are not initialized.
            continue # TODO wtf?
         else:
            #print "adding",c,scores[c]/gnormalizer
            self.add(features, c, self.scores[c]/gnormalizer)
      for c in xrange(self.nclasses):
         if self.scores[c] > 0:
         #print "adding",c,-scores[c]/normalizer
            self.add(features, c, -self.scores[c])

   cpdef add(self, list features, int clas, double amount):
      """
      RDA update based on:
         http://code.google.com/p/factorie/source/browse/src/main/scala/cc/factorie/optimize/AdaGradRDA.scala
      l1 is not working.
      """
      #self.rate = 10
      #cdef double l1 = 0.01
      cdef SparseMulticlassParamData p
      #cdef double to_add
      #cdef double delta = 0.1
      #cdef double h
      #cdef double acc
      for f in features:
         try:
            p = self.W[f]
            #g = self.G[f]
         except KeyError:
            p = SparseMulticlassParamData()
            #g = SparseMulticlassParamData()
            self.W[f] = p
            #self.G[f] = g
         # calculate the addition:
         p.add_to_clas_acc2(clas, (1*amount)*(1*amount))
         p.add_to_clas_acc(clas, 1*amount)
         #h = sqrt(p.acc2_for_clas(clas)) + delta
         #to_add = 1.0 / ((1.0 / self.rate) * h)
         #acc = p.acc_for_clas(clas)
         #print "acc",acc,l1*self.now
         #if acc >= 0:
         #   acc -= (l1 * self.now)
         #   if acc > 0:
         #      #print "up1:",to_add*acc
         #      p.set_clas_w_to(clas, to_add * acc, self.now, 0)
         #else:
         #   acc += (l1 * self.now)
         #   if acc < 0:
         #      #print "up2:",to_add*acc
         #      p.set_clas_w_to(clas, to_add * acc, self.now, 0)
         # all feature weights are 1, and 1^2 = 1

   cpdef get_scores(self, features):
      cdef SparseMulticlassParamData p
      cdef int c
      cdef double w
      cdef double h
      cdef double rate = self.rate
      cdef double delta = 0.01 #self.delta
      cdef double l1 = 0.01/10000 #self.l2
      cdef double l2 = 0.0 #self.l2
      cdef double t1
      cdef double grads
      for c in xrange(self.nclasses):
         self.scores[c]=0
      for f in features:
         try:
            p = self.W[f]
            for c in xrange(self.nclasses):
               h = (1.0/rate) * (sqrt(p.acc2_for_clas(c)) + delta) + (self.now * l2)
               t1 = 1.0 / h
               grads = p.acc_for_clas(c)
               if (grads > self.now * l1):
                  t1 = t1 * (grads - (self.now * l1))
               elif (grads < -(self.now * l1)):
                  t1 = t1 * (grads + (self.now * l1))
               else:
                  t1 = 0.0
               self.scores[c] += t1
         except KeyError: pass
      res={}
      for c in xrange(self.nclasses):
         res[c] = self.scores[c]
      return res

   cdef _get_normed_exp_scores(self, features, double* scores):
      """
      see: http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
      for explanation about the "a" subtraction.
      """
      cdef SparseMulticlassParamData p
      cdef int i
      cdef double w
      cdef double e
      cdef double a
      for i in xrange(self.nclasses):
         scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            p.add_w_to_scores(scores)
         except KeyError: pass
      cdef double tot = 0
      a = self.scores[0]
      for i in xrange(self.nclasses):
         if scores[i] > a:
            a = scores[i]
         
      for i in xrange(self.nclasses):
         e = exp(scores[i] - a)
         scores[i] = e
         tot += e
      for i in xrange(self.nclasses):
         scores[i] = scores[i] / tot

   cpdef get_normed_exp_scores(self, features):
      self._get_normed_exp_scores(features, self.scores)
      res={}
      for i in xrange(self.nclasses):
         res[i] = self.scores[i]
      return res

      #cdef SparseMulticlassParamData p
      #cdef int i
      #cdef double w
      #cdef double e
      #cdef double a
      #for i in xrange(self.nclasses):
      #   self.scores[i]=0
      #for f in features:
      #   try:
      #      p = self.W[f]
      #      p.add_w_to_scores(self.scores)
      #   except KeyError: pass
      #cdef double tot = 0
      #res={}
      #a = self.scores[0]
      #for i in xrange(self.nclasses):
      #   if self.scores[i] > a:
      #      a = self.scores[i]
      #   
      #for i in xrange(self.nclasses):
      #   e = exp(self.scores[i] - a)
      #   self.scores[i] = e
      #   tot += e
      #for i in xrange(self.nclasses):
      #   res[i] = self.scores[i] / tot
      #   #print "s,e(s):",i,self.scores[i],math.exp(self.scores[i])
      #   #res[i] = math.exp(self.scores[i]) / tot
      #return res

   def finalize(self):
      pass
      #cdef SparseMulticlassParamData p
      # average
      #for f in self.W.keys():
      #   p = self.W[f]
      #   p.finalize(self.now)

   def dump(self, out=sys.stdout, sparse=False):
      cdef SparseMulticlassParamData p
      cdef int c
      cdef double w
      cdef double h
      cdef double rate = self.rate
      cdef double delta = 0.01 #self.delta
      cdef double l1 = 0.01 / 10000 #self.l2
      cdef double l2 = 0.0 #self.l2
      cdef double t1
      cdef double grads
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         p = self.W[f]
         out.write("%s" % f)
         for c in xrange(self.nclasses):
            h = (1.0/rate) * (sqrt(p.acc2_for_clas(c)) + delta) + (self.now * l2)
            t1 = 1.0 / h
            grads = p.acc_for_clas(c)
            if (grads > self.now * l1):
               t1 = t1 * (grads - (self.now * l1))
            elif (grads < -(self.now * l1)):
               t1 = t1 * (grads + (self.now * l1))
            else:
               t1 = 0.0
            if sparse:
               if t1 != 0:
                  out.write(" %s:%s" % (c,t1))
            else:
               out.write(" %s" % t1)
         out.write("\n")

   #def dump_fin(self,out=sys.stdout, sparse=False):
   #   cdef SparseMulticlassParamData p
   #   # write the average
   #   if sparse:
   #      out.write("%s\n" % self.nclasses)
   #   for f in self.W.keys():
   #      out.write("%s" % f)
   #      for c in xrange(self.nclasses):
   #         p = self.W[f]
   #         w = p.avgd_w_for_clas(c, self.now)
   #         if sparse:
   #            if w != 0:
   #               out.write(" %s:%s" % (c,w))
   #         else:
   #            out.write(" %s " % (w))
   #      out.write("\n")

#}}}

