import spacy, alg
from scipy import sparse
from alg import dphs, dtls, outphs, nlp
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

print "Loading complete."

def glem(x):
    d = nlp(unicode(x))
    return " ".join((x.lemma_ for x in d
                    if not x.is_stop and x.is_alpha and len(x.text) > 1))


try:
    d_xtr = []
    for title in dtls.title.tolist():    
        b = alg.inxter(title,)
        b = alg.get_phs(*b)
        d_xtr.append("".join(b[:-1].rtw))
except IndexError, ex:
    print "Warning: %s\r\n" % ex.message
   

v = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', preprocessor=glem, min_df=8, max_df=.9, norm="l2", stop_words=spacy.en.STOP_WORDS).fit(d_xtr)
d_xtrtr = v.transform(d_xtr)

def get_answer(q, upr=10):
    temp_results = dtls.iloc[cosine_distances(v.transform([q]), d_xtrtr)[0].argsort()[:10]].title.tolist()
    topresult = "".join(
        alg.get_phs(*alg.inxter(temp_results[0]))[:-1]\
        .apply(lambda x: "<p style=\"font-size:%dpx;font-family:Century Gothic;\">"
               % int(x.fqs * (1.15 if x.fqs > 16 else 1.))
               + ("<b>" if "bold" in x.fqf.lower() else "")
               + x.rtw
               + ("</b>" if "bold" in x.fqf.lower() else "")
               + "</p>", axis=1))
    if upr == 1:
        return topresult
    return [topresult] + ["".join(
        alg.get_phs(*alg.inxter(resix)).pipe(lambda x: x[:min(5, x.shape[0])])\
        .apply(lambda x: "<p style=\"font-size:%dpx;font-family:Century Gothic;\">"
               % int(x.fqs * (1.15 if x.fqs > 16 else 1.))
               + ("<b>" if "bold" in x.fqf.lower() else "")
               + x.rtw
               + ("</b>" if "bold" in x.fqf.lower() else "")
               + "</p>", axis=1)) for resix in temp_results[1:min(upr, 10)]]
