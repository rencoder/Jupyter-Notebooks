
import argparse
from scipy.stats import hmean
import alg_img
from string import punctuation
from flask import Flask, request, jsonify, send_from_directory
import module1, module2
import pandas as pd


nlp = module1.nlp
DOC_M = [module1, module2]
DF_OUTLINES = pd.DataFrame()
for item in DOC_M:
    DF_OUTLINES = pd.concat([DF_OUTLINES, item.obtain_outlines().assign(doc_name=item.__name__)])


def get_answer(q):
    q = " ".join([x.strip(punctuation).lower() for x in q.split(" ")])
    q = nlp(unicode(q) if not isinstance(q, unicode) else q)
    q = nlp(u" ".join([(x.lemma_ if x.text not in ["data"] else x.text)
                       for x in q if not (x.is_stop or x.lower_ in ["bot", "help"])]))
    q_set = set(q.text.split())
    def safe_hmean(citem):
        try:
            return hmean((citem.iloc[0], citem.iloc[1]))
        except:
            return 0.
    results = DF_OUTLINES.assign(
        treated=lambda x: x.title.str.apply(unicode).apply(
            lambda x: nlp(u" ".join([(w.lemma_ if w.text not in ["data"] else w.text)
                                     for w in nlp(x.strip()) if not w.is_stop])))).assign(
        sc_sim=lambda x: x.treated.apply(lambda y: q.similarity(y))).assign(
        sc_eq=lambda x: x.treated.apply(lambda x: x.text).str.split(" ").apply(
            lambda y: float(len(q_set.intersection(y))) / max(len(q_set), len(y)))).assign(
        sc_total=lambda x: x[["sc_sim", "sc_eq"]].apply(safe_hmean, axis=1)).sort_values(
        ["sc_total", "sc_eq"], ascending=[0, 0]).drop_duplicates(
        subset="title").iloc[:10][["title", "sc_total", "pageno", "doc_name"]].copy(deep=True)
    module = eval(results.iloc[0].doc_name)
    b = module.get_bounds(results.iloc[0].title)
    rpages = module.get_lines_paged(*b)
    jawab = {
        "pc": [{
            "pageno": int(groupname),
            "content": "".join(groupdf.apply(module.md_formatter, axis=1))}
            for (groupname, groupdf) in rpages],
        "q": q.text,
        "filename": module.path,
        "orp": []
    }
    for i in range(1, 6):
        module = eval(results.iloc[i].doc_name)
        b = module.get_bounds(results.iloc[i].title)
        rpages = module.get_lines(*b)
        jawab["orp"].append({
            "pageno": int(rpages.iloc[0].values[0]),
            "filename": module.path,
            "previewText": "".join(rpages.iloc[1:min(rpages.shape[0], 6)].raw_text.values.tolist()),
            "title": rpages.iloc[0].raw_text
        })
    return jawab


app = Flask("app1")


@app.route("/question", methods=["POST"])
def getanswer():
    data = request.get_json(force=True)
    jawab = get_answer(data["input_text"]) if data["doc_set"] != "modules" else alg_img.get_answer(data["input_text"])
    return jsonify({
        "filename": jawab.get("filename") or alg_img.path,
        "final_query": jawab.get("q"),
        "page_content": jawab.get("pc"),
        "isValid": bool(jawab.get("pc")),
        "other_relevant_pages": jawab.get("orp")
    })


@app.route("/<string:dir_name>/<path:filename>")
def download_image(dir_name, filename):
    return send_from_directory(dir_name, filename)


parser = argparse.ArgumentParser()
parser.add_argument("--port")
port = parser.parse_args().port or 1903
app.run("0.0.0.0", port=int(port))
