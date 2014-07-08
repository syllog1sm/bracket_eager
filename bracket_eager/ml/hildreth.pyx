"""
Hildreth qp algorithm.
Based on the Java implementation in Ryan McDonald's MSTparser.

Quick and Dirty, nothing optimized.

Supports up to 3000 constraints (hard-limit to avoid manual memory management, 
easy to extend by s/3000/300/g )

Yoav Goldberg 2011 (first.last@gmail.com)
"""
from stdlib cimport malloc, free

cdef class Integer:
   cdef int i
   def __init__(self, int i):
      self.i=i
   def value(self): return self.i

cdef class FV:
   cdef dict fv
   def __init__(self, list pos, list neg):
      cdef Integer i
      self.fv={}
      for f in pos:
         if f in self.fv:
            i=self.fv[f]
            i.i = i.i + 1
         else: self.fv[f] = Integer(1)
      for f in neg:
         if f in self.fv:
            i = self.fv[f]
            i.i = i.i - 1
         else:
            self.fv[f] = Integer(-1)

   cdef double dot(self, FV other):
      cdef Integer v1
      cdef Integer v2
      cdef double s=0
      for f,v1 in self.fv.iteritems():
         if f in other.fv:
            v2 = other.fv[f]
            s+=v1.i*v2.i
      return s


cpdef hildreth(fs, bs):
   """
   The way I usually represent feature-vectors in my python code is as lists of strings.
   Each string is an indicator feature with a value of 1, and if the same string appears twice
   in the list, its value is 2. This representation allows positive count features.
   In order to allow negative count features as well, I use a tuple of lists (pos,neg) where
   the semantics is to subtract the vector neg from the vector pos.

   fs: a list of k (pos,neg) feature-vec pairs [(pos1,neg1),(pos2,neg2),...,(posk,negk)]
   bs: a list of k floats
   ###the constraints are alpha_i*fs[i]<bs[i]
   the constraints are alpha_i*fs[i]>bs[i]
   """
   assert(len(fs)==len(bs))
   cdef int lenb = len(bs)
   if lenb>=3000: raise ValueError("too many constraints (%s), not supported." % lenb)
   cdef double[3000] b
   cdef int i = 0
   for _b in bs:
      b[i] = _b
      i+=1
   FVs = []
   for pos,neg in fs:
      FVs.append(FV(pos,neg))
   res = _hildreth(FVs, b, lenb)
   return res

cdef list _hildreth(list a, double* b, int lenb):
   cdef int i, j
   cdef int max_iter = 10000
   cdef double eps = 0.00000001
   cdef double zero = 0.000000000001

   cdef double[3000] alpha

   cdef double[3000] F
   cdef double[3000] kkt


   cdef double max_kkt = -999999

   cdef int K=lenb

   #cdef double[3000*3000] A
   cdef double* A = <double*>malloc(sizeof(double)*3000*3000)
   for i in xrange(K):
      for j in xrange(K):
         A[i+(3000*j)]=0

   for i in xrange(K): 
      alpha[i]=0
      F[i]=0
      kkt[i]=0

   cdef short[3000] is_computed
   for i in xrange(3000): is_computed[i]=0

   cdef FV ai 
   cdef FV aj
   for i in xrange(K):
      ai = a[i]
      A[i+(3000*i)] = ai.dot(ai)
      is_computed[i] = 0

      F[i] = b[i]
      kkt[i] = F[i]

   max_kkt = kkt[0]
   cdef int max_kkt_i = 0

   for i in xrange(K):
      if (kkt[i] > max_kkt):
         max_kkt = kkt[i]
         max_kkt_i = i

   cdef int itr = 0
   cdef double diff_alpha
   cdef double try_alpha
   cdef double add_alpha

   while max_kkt >= eps and itr < max_iter:
      #print "hildreth iter:",itr,max_kkt
      diff_alpha = 0.0 if A[max_kkt_i+(3000*max_kkt_i)] <= zero else F[max_kkt_i]/A[max_kkt_i+(3000*max_kkt_i)]
      try_alpha  = alpha[max_kkt_i] + diff_alpha
      add_alpha  = 0
      if try_alpha < 0:
         add_alpha = -1.0 * alpha[max_kkt_i]
      else:
         add_alpha = diff_alpha

      alpha[max_kkt_i] = alpha[max_kkt_i] + add_alpha

      if (0 != is_computed[max_kkt_i]):
         for i in xrange(K):
            ai = a[i]
            aj = a[max_kkt_i]
            A[i+(3000*max_kkt_i)] = ai.dot(aj)
            is_computed[max_kkt_i]=1

      for i in xrange(K):
         F[i] = F[i] - add_alpha*A[i+(3000*max_kkt_i)]
         kkt[i] = F[i]
         if alpha[i] > zero:
            kkt[i] = F[i] if F[i]>0 else -F[i]

      max_kkt = kkt[0]
      max_kkt_i = 0
      for i in xrange(K):
         if kkt[i] > max_kkt:
            max_kkt = kkt[i]
            max_kkt_i = i
      itr+=1
   
   free(A)
   if itr == max_iter: return []

   res=[]
   for i in xrange(K):
      res.append(alpha[i])
   return res




