gradient(features, goods, model):
   scores = model.get_linear_scores(features)
   # assuming scores is a dict from class to score
   scores = [(clas,exp(s)) for clas,s in scores.iteritems()]
   normalizer = sum([s for c,s in scores])
   predictions = [(clas,s/normalizer) for clas,s in scores]

   goods = [(clas,s) for clas,s in scores if clas in goods]
   gnormalizer = sum([s for c,s in goods])
   gpredictions = [(clas,s/gnormalizer) for clas,s in goods]

   grad = []
   g = grad.append
   for c,s in gpredictions:
      for f in features:
         g((f,c,s))
   for c,s in predictions:
      for f in features:
         g((f,c,-s))
   return grad

