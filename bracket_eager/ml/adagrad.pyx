import sys
from libc.stdlib cimport *
from libc.math cimport exp, sqrt
import cython

cdef class AdagradParamData:
   cdef:
      float *acc
      float *w
      int   *last_update
   def __cinit__(self, int nclasses):
      cdef int i
      self.acc     = <float *>malloc(nclasses*sizeof(float))
      self.w       = <float *>malloc(nclasses*sizeof(float))
      self.last_update = <int *>malloc(nclasses*sizeof(int))
      for i in xrange(nclasses):
         self.acc[i]=0
         self.w[i]=0
         self.last_update[i]=-1

   def __dealloc__(self):
      free(self.acc)
      free(self.w)
      free(self.last_update)

cdef double truncate(double x, double l):
   if x > l: return x - l
   elif x < -l: return x + l
   else: return 0.0

cdef class L1RegularizedAdagradParameters: #{{{
   cdef:
      int nclasses
      int now
      double rate
      double l1
      int ygreg
      dict W
      double* scores # (re)used in calculating prediction

   def __cinit__(self, nclasses, double rate=1.0, double l1=0.0, int ygreg=0):
      self.scores = <double *>malloc(nclasses*sizeof(double))
      self.nclasses = nclasses
      self.W = {}
      self.rate = rate
      self.now = 0
      self.l1 = l1
      #This is the "sparse perceptron" regularization.
      #NOTE: assume all vectors are always binary, and all updates are 1/-1 also.
      self.ygreg = ygreg

   cdef _tick(self):
      self.now=self.now+1

   def tick(self):
      self._tick()

   def num_params(self): return -1

   @cython.cdivision(True)
   cdef regularize(self, list features):
      cdef AdagradParamData p
      cdef double delta = 0.1
      cdef double l1 = self.l1
      cdef int c
      cdef float w
      cdef float h
      cdef double panelty
      cdef int elapsed
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            continue
         for c in xrange(self.nclasses):
            if p.last_update[c] == -1: continue
            elapsed = self.now - p.last_update[c]
            h = sqrt(p.acc[c]) + delta
            panelty = elapsed * l1 * (self.rate / h)
            w = p.w[c]
            p.w[c] = truncate(w,panelty)
            p.last_update[c] = self.now

   @cython.cdivision(True)
   cpdef update_me(self, list features, list good_classes):
      self._get_normed_exp_scores(features, self.scores)
      cdef double gnormalizer = 0
      cdef int c
      cdef double s

      for c in good_classes:
         gnormalizer += self.scores[c]

      for c in good_classes:
         if gnormalizer == 0:
            self.add(features, c, 0) # add 0 instead of skipping to initialize the data structures if they are not initialized.
            continue # TODO wtf?
         else:
            self.add(features, c, self.scores[c]/gnormalizer)
      for c in xrange(self.nclasses):
         if self.scores[c] > 0:
            self.add(features, c, -self.scores[c])

   @cython.cdivision(True)
   cpdef add(self, list features, int clas, float amount):
      cdef AdagradParamData p
      cdef double to_add
      cdef double delta = 0.1
      cdef double h
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = AdagradParamData(self.nclasses)
            self.W[f] = p
         # calculate the addition:
         p.acc[clas] += (1*amount)*(1*amount)
         h = sqrt(p.acc[clas]) + delta
         to_add = (1*amount) * (self.rate / h)
         p.w[clas] += to_add
         if p.last_update[clas] == -1: p.last_update[clas] = self.now

   cdef _get_scores(self, list features, double *scores):
      self.regularize(features)
      cdef AdagradParamData p
      cdef int i
      cdef double w
      for i in xrange(self.nclasses):
         scores[i]=0
      for f in features:
         try:
            p = self.W[f]
            for i in xrange(self.nclasses):
               if p.acc[i] > self.ygreg:
                  scores[i] += p.w[i]
         except KeyError: pass

   cpdef get_scores(self, list features):
      cdef int i
      self._get_scores(features, self.scores)
      return {i : self.scores[i] for i in xrange(self.nclasses)}

   @cython.cdivision(True)
   cdef _get_normed_exp_scores(self, list features, double* scores):
      """
      see: http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
      for explanation about the "a" subtraction.
      """
      cdef double tot
      cdef double a
      cdef double e
      cdef int i
      self._get_scores(features, scores)
      a = scores[0]
      for i in xrange(self.nclasses):
         if scores[i] > a:
            a = scores[i]
         
      for i in xrange(self.nclasses):
         e = exp(scores[i] - a)
         scores[i] = e
         tot += e
      for i in xrange(self.nclasses):
         scores[i] = scores[i] / tot

   cpdef get_normed_exp_scores(self, list features):
      cdef int i
      self._get_normed_exp_scores(features, self.scores)
      return {i : self.scores[i] for i in xrange(self.nclasses)}

   def dump(self, out=sys.stdout, sparse=False):
      cdef AdagradParamData p
      cdef int c
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         out.write("%s" % f)
         p = self.W[f]
         for c in xrange(self.nclasses):
            if sparse:
               if p.w[c] != 0 and p.acc[c] > self.ygreg:
                  out.write(" %s:%s" % (c,p.w[c]))
            else:
               if self.ygreg > 0: assert(False),"not supported"
               out.write(" %s" % p.w[c])
         out.write("\n")

   def finalize(self):
      self.regularize(self.W.keys())

   def dump_fin(self,out=sys.stdout, sparse=False):
      self.regularize(self.W.keys())
      self.dump(out, sparse)
#}}}

cdef struct RDAParam: #{{{
   double acc
   double acc2
#}}}

cdef class SparseMulticlassParamData: #{{{
   cdef RDAParam *params
   cdef int current_size
   cdef int *clasmap
   def __cinit__(self):
      cdef int i
      cdef int initial_size = 2
      self.params = <RDAParam *>malloc(initial_size*sizeof(RDAParam))
      self.clasmap = <int *>malloc(initial_size*sizeof(int))
      self.current_size = initial_size
      for i in range(self.current_size):
         self.clasmap[i]=-1
         self.params[i].acc=0
         self.params[i].acc2=0
   
   cdef int clas_idx(self, int clas):
      cdef int i
      for i in xrange(self.current_size):
         if self.clasmap[i] == -1:
            return -1
         elif self.clasmap[i] == clas:
            return i
      return -1

   cdef double acc2_for_clas(self, int clas):
      cdef int idx
      idx = self.clas_idx(clas)
      if idx < 0: return 0
      return self.params[idx].acc2

   cdef double acc_for_clas(self, int clas):
      cdef int idx
      idx = self.clas_idx(clas)
      if idx < 0: return 0
      return self.params[idx].acc

   cdef int get_param_index_for(self, int clas):
      """
      Get the index of the Param for the given clas,
      creating a new one if needed.
      """
      cdef int i
      for i in xrange(self.current_size):
         if self.clasmap[i] == clas:
            return i
         elif self.clasmap[i] == -1:
            self.clasmap[i] = clas
            return i
      # If we got here, we scanned all entries without
      # finding the clas. Increase the size and add the clas.
      cdef int new_size = self.current_size * 2
      cdef RDAParam *new_params = <RDAParam *>malloc(new_size*sizeof(RDAParam))
      cdef int *new_clasmap = <int *>malloc(new_size*sizeof(int))
      for i in xrange(self.current_size):
         new_params[i] = self.params[i]
         new_clasmap[i] = self.clasmap[i]
      free(self.params)
      free(self.clasmap)
      self.params = new_params
      self.clasmap = new_clasmap
      for i in xrange(self.current_size, new_size):
         self.clasmap[i]=-1
         self.params[i].acc=0
         self.params[i].acc2=0
      self.clasmap[self.current_size] = clas
      cdef int res = self.current_size
      self.current_size = new_size
      return res

   def __dealloc__(self):
      free(self.params)
      free(self.clasmap)
#}}}

cdef class DenseMulticlassParamData: #{{{
   """
   Dense in terms of labels.
   """
   cdef RDAParam *params
   cdef int current_size
   def __cinit__(self, int nclasses):
      cdef int i
      cdef int initial_size = nclasses
      self.params = <RDAParam *>malloc(initial_size*sizeof(RDAParam))
      self.current_size = initial_size
      for i in range(self.current_size):
         self.params[i].acc=0
         self.params[i].acc2=0
   
   cdef int clas_idx(self, int clas):
      return clas

   cdef double acc2_for_clas(self, int clas):
      cdef int idx
      idx = self.clas_idx(clas)
      if idx < 0: return 0
      return self.params[idx].acc2

   cdef double acc_for_clas(self, int clas):
      cdef int idx
      idx = self.clas_idx(clas)
      if idx < 0: return 0
      return self.params[idx].acc

   cdef int get_param_index_for(self, int clas):
      """
      Get the index of the Param for the given clas,
      creating a new one if needed. /NOT CREATING. THEY ARE ALL THERE
      """
      return clas

   def __dealloc__(self):
      free(self.params)
#}}}

cdef class SparseRDAAdagradParameters: #{{{
   """
   RDA-regularized AdaGrad, with sparse-labels.
   """
   cdef:
      int nclasses
      int now
      double rate
      dict W
      double* scores # (re)used in calculating prediction
      double l1
      double l2
   
   def __cinit__(self, int nclasses, double rate=1.0, double l1=0.0):
      self.scores = <double *>malloc(nclasses*sizeof(double))
      self.nclasses = nclasses
      self.W = {}
      self.rate = rate
      self.now = 0
      self.l1 = l1
      self.l2 = 0.0

   cdef _tick(self):
      self.now=self.now+1

   def tick(self): 
      self._tick()

   def num_params(self): return -1

   @cython.cdivision(True)
   cpdef update_me(self, list features, list good_classes):
      self._get_normed_exp_scores(features, self.scores)
      cdef double gnormalizer = 0
      cdef int c
      cdef double s

      for c in good_classes:
         gnormalizer += self.scores[c]

      for c in good_classes:
         if gnormalizer == 0:
            self.add(features, c, 0) # add 0 instead of skipping to initialize the data structures if they are not initialized.
            continue # TODO wtf?
         else:
            self.add(features, c, self.scores[c]/gnormalizer)
      for c in xrange(self.nclasses):
         if self.scores[c] > 0:
            self.add(features, c, -self.scores[c])

   @cython.cdivision(True)
   cpdef update_percep(self, list features, list good_classes):
      self._get_scores(features, self.scores)
      cdef int c
      cdef double s

      cdef int best_c = 0
      cdef double max_c = self.scores[best_c]
      for c in xrange(self.nclasses):
         if self.scores[c] > max_c:
            max_c = self.scores[c]
            best_c = c

      cdef int best_gc = good_classes[0]
      cdef double max_gc = self.scores[best_gc]
      for c in good_classes:
         if self.scores[c] > max_c:
            max_gc = self.scores[c]
            best_gc = c

      if best_gc != best_c:
         self.add(features, best_gc, 1) 
         self.add(features, best_c, -1)

   cpdef add(self, list features, int clas, double amount):
      """
      RDA update based on:
         http://code.google.com/p/factorie/source/browse/src/main/scala/cc/factorie/optimize/AdaGradRDA.scala
      """
      cdef SparseMulticlassParamData p
      cdef int idx
      if amount == 0: return
      for f in features:
         p = self.W.get(f,None)
         if p == None:
            p = SparseMulticlassParamData()
            self.W[f] = p
         # calculate the addition:
         idx = p.get_param_index_for(clas)
         p.params[idx].acc += (1*amount)
         p.params[idx].acc2 += (1*amount)*(1*amount)
         
   @cython.cdivision(True)
   cdef _get_scores(self, list features, double *scores):
      cdef SparseMulticlassParamData p
      cdef int c
      cdef double w
      cdef double h
      cdef double rate = self.rate
      cdef double delta = 0.01 #self.delta
      cdef double t1
      cdef double grads
      cdef int idx
      for c in xrange(self.nclasses):
         scores[c]=0
      for f in features:
         try:
            p = self.W[f]
            for c in xrange(self.nclasses):
               idx = p.clas_idx(c)
               if idx < 0: continue
               if p.params[idx].acc2 == 0: continue
               h = (1.0/rate) * (sqrt(p.params[idx].acc2) + delta) + (self.now * self.l2)
               t1 = 1.0/h
               scores[c] += t1 * truncate(p.params[idx].acc, self.now * self.l1)
         except KeyError: pass

   cpdef get_scores(self, list features):
      cdef int i
      self._get_scores(features, self.scores)
      return {i : self.scores[i] for i in xrange(self.nclasses)}

   @cython.cdivision(True)
   cdef _get_normed_exp_scores(self, list features, double* scores):
      """
      see: http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
      for explanation about the "a" subtraction.
      """
      cdef double a
      cdef double e
      cdef double tot = 0
      cdef int i
      self._get_scores(features, scores)
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

   cpdef get_normed_exp_scores(self, list features):
      cdef int i
      self._get_normed_exp_scores(features, self.scores)
      return {i : self.scores[i] for i in xrange(self.nclasses)}
   

   def finalize(self):
      pass

   @cython.cdivision(True)
   def dump(self, out=sys.stdout, sparse=False):
      cdef SparseMulticlassParamData p
      cdef int c
      cdef double w
      cdef double h
      cdef double rate = self.rate
      cdef double delta = 0.01 #self.delta
      cdef double t1
      cdef double grads
      cdef int idx
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         p = self.W[f]
         for c in xrange(self.nclasses):
            self.scores[c] = 0
            idx = p.clas_idx(c)
            if idx < 0: continue
            if p.params[idx].acc2 == 0: continue
            h = (1.0/rate) * (sqrt(p.params[idx].acc2) + delta) + (self.now * self.l2)
            t1 = 1.0/h
            self.scores[c] += t1 * truncate(p.params[idx].acc, self.now * self.l1)
         if sparse:
            all_zeros = True
            for c in xrange(self.nclasses):
               if self.scores[c] != 0:
                  all_zeros = False
                  break
            if not all_zeros:
               out.write("%s" % f)
               for c in xrange(self.nclasses):
                  if self.scores[c] != 0:
                     out.write(" %s:%s" % (c,self.scores[c]))
               out.write("\n")
         else:
            out.write("%s" % f)
            for c in xrange(self.nclasses): out.write(" %s" % self.scores[c])
            out.write("\n")
#}}}

cdef class DenseRDAAdagradParameters: #{{{
   """
   RDA-regularized AdaGrad, with dense-labels.
   """
   #TODO
   cdef:
      int nclasses
      int now
      double rate
      dict W
      double* scores # (re)used in calculating prediction
      double l1
      double l2
   
   def __cinit__(self, int nclasses, double rate=1.0, double l1=0):
      self.scores = <double *>malloc(nclasses*sizeof(double))
      self.nclasses = nclasses
      self.W = {}
      self.rate = rate
      self.now = 0
      self.l1 = l1
      self.l2 = 0.0

   cdef _tick(self):
      self.now=self.now+1

   def tick(self): 
      self._tick()

   @cython.cdivision(True)
   cpdef update_me(self, list features, list good_classes):
      self._get_normed_exp_scores(features, self.scores)
      cdef double gnormalizer = 0
      cdef int c
      cdef double s

      for c in good_classes:
         gnormalizer += self.scores[c]

      for c in good_classes:
         if gnormalizer == 0:
            self.add(features, c, 0) # add 0 instead of skipping to initialize the data structures if they are not initialized.
            continue # TODO wtf?
         else:
            self.add(features, c, self.scores[c]/gnormalizer)
      for c in xrange(self.nclasses):
         if self.scores[c] > 0:
            self.add(features, c, -self.scores[c])

   cpdef add(self, list features, int clas, double amount):
      """
      RDA update based on:
         http://code.google.com/p/factorie/source/browse/src/main/scala/cc/factorie/optimize/AdaGradRDA.scala
      """
      cdef DenseMulticlassParamData p
      cdef int idx
      for f in features:
         try:
            p = self.W[f]
         except KeyError:
            p = DenseMulticlassParamData(self.nclasses)
            self.W[f] = p
         # calculate the addition:
         idx = p.get_param_index_for(clas)
         p.params[idx].acc += (1*amount)
         p.params[idx].acc2 += (1*amount)*(1*amount)
         
   @cython.cdivision(True)
   cdef _get_scores(self, list features, double *scores):
      cdef DenseMulticlassParamData p
      cdef int c
      cdef double w
      cdef double h
      cdef double rate = self.rate
      cdef double delta = 0.01 #self.delta
      cdef double t1
      cdef double grads
      cdef int idx
      for c in xrange(self.nclasses):
         scores[c]=0
      for f in features:
         try:
            p = self.W[f]
            for c in xrange(self.nclasses):
               idx = p.clas_idx(c)
               if idx < 0: continue
               if p.params[idx].acc2 == 0: continue
               h = (1.0/rate) * (sqrt(p.params[idx].acc2) + delta) + (self.now * self.l2)
               t1 = 1.0/h
               scores[c] += t1 * truncate(p.params[idx].acc, self.now * self.l1)
         except KeyError: pass

   cpdef get_scores(self, list features):
      self._get_scores(features, self.scores)
      return {i : self.scores[i] for i in xrange(self.nclasses)}

   @cython.cdivision(True)
   cdef _get_normed_exp_scores(self, list features, double* scores):
      """
      see: http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
      for explanation about the "a" subtraction.
      """
      cdef double a
      cdef double e
      cdef double tot = 0
      cdef int i
      self._get_scores(features, scores)
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

   cpdef dict get_normed_exp_scores(self, list features):
      cdef int i
      self._get_normed_exp_scores(features, self.scores)
      return {i : self.scores[i] for i in xrange(self.nclasses)}

   def finalize(self):
      pass

   @cython.cdivision(True)
   def dump(self, out=sys.stdout, sparse=False):
      cdef DenseMulticlassParamData p
      cdef int c
      cdef double w
      cdef double h
      cdef double rate = self.rate
      cdef double delta = 0.01 #self.delta
      cdef double t1
      cdef double grads
      cdef int idx
      if sparse:
         out.write("%s\n" % self.nclasses)
      for f in self.W.keys():
         p = self.W[f]
         for c in xrange(self.nclasses):
            self.scores[c] = 0
            idx = p.clas_idx(c)
            if idx < 0: continue
            if p.params[idx].acc2 == 0: continue
            h = (1.0/rate) * (sqrt(p.params[idx].acc2) + delta) + (self.now * self.l2)
            t1 = 1.0/h
            self.scores[c] += t1 * truncate(p.params[idx].acc, self.now * self.l1)
         if sparse:
            all_zeros = True
            for c in xrange(self.nclasses):
               if self.scores[c] != 0:
                  all_zeros = False
                  break
            if not all_zeros:
               out.write("%s" % f)
               for c in xrange(self.nclasses):
                  if self.scores[c] != 0:
                     out.write(" %s:%s" % (c,self.scores[c]))
               out.write("\n")
         else:
            out.write("%s" % f)
            for c in xrange(self.nclasses): out.write(" %s" % self.scores[c])
            out.write("\n")
#}}}

