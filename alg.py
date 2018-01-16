import re, spacy
import numpy as np
import pandas as pd
import pdfminer
from gc import collect
from collections import Counter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
from IPython.display import HTML, display


def convert_pdf(path, dest="text"):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = "utf-8"
    laparams = LAParams()
    device = (TextConverter if dest=="text" else XMLConverter)(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, "rb")
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    btype="xml"
    maxpages = 0
    caching = True
    pagenos = set()
    from bs4 import BeautifulSoup as bs
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    s = retstr.getvalue()
    s = bs(s, "xml")
    retstr.close()
    return s

nlp = spacy.load("en")
r = convert_pdf(path, "x")

def get_outphs(path):
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.psparser import PSLiteral
    from pdfminer.pdftypes import resolve1
    fp = open(path, "rb")
    parser = PDFParser(fp)
    document = PDFDocument(parser)
    pages = dict((page.pageid, pageno) for (pageno,page)
              in enumerate(PDFPage.create_pages(document)))
    def resolve_dest(dest):
        if isinstance(dest, str):
            dest = resolve1(document.get_dest(dest))
        elif isinstance(dest, PSLiteral):
            dest = resolve1(document.get_dest(dest.name))
        if isinstance(dest, dict):
            dest = dest['D']
        return dest
    toc = []
    for (level, title, dest, a, structelem) in document.get_outlines():
        pageno = None
        if dest:
            dest = resolve_dest(dest)
            pageno = pages[dest[0].objid]
        elif a:
            action = a
            if isinstance(action, dict):
                subtype = action.get('S')
                if subtype and repr(subtype) == '/GoTo' and action.get('D'):
                    dest = resolve_dest(action['D'])
                    pageno = pages[dest.resolve()[0].objid]
        toc.append({"level": level, "raw_title": title, "pageno": pageno + 1})
    return toc
    

dtls = pd.DataFrame(get_outphs(path))
dtls.loc[:, "title"] = dtls.raw_title.str.replace("\n", " ").str.strip().str.lower()
outphs = dtls.title.tolist()


phs = []
for l in r.find_all("textline"):
    d = dict.fromkeys(["text", "fonts", "sizes"])
    texts = l.find_all("text")
    text = "".join([x.text for x in texts])#.strip()
    if not text: continue
    d["text"] = text.strip()
    d["raw_text"] = text
    d["fonts"] = [x["font"] for x in texts if x.has_attr("font")]
    d["sizes"] = [round(np.float(x.get("size") or .0)) for x in texts]
    d["pageno"] = int(l.findParent("page")["id"])
    phs.append(d)


df_phs = pd.DataFrame.from_dict(phs).assign(fqs=lambda x: x["sizes"].apply(lambda y: Counter(y).most_common(1)[0][0])).assign(fqf=lambda x: x["fonts"].apply(lambda y: Counter(y).most_common(1)[0][0])).assign(text=lambda x: x.raw_text.str.replace("\n", " ").str.strip().str.lower()).drop(["fonts", "sizes"], 1)
levels = dtls.level.tolist()


def get_bounds(q):
    startix, level = dtls[dtls.title.str.startswith(q)].pipe(lambda x: (x.index[0], x.level.iloc[0]))
    endix = startix + 1
    while levels[endix] > level:
        endix += 1
    return dtls.loc[[startix, endix]][["title", "pageno"]].values.tolist()


def get_phs(start, end):
    dfmatch = pd.DataFrame()
    def match(q):
        for _ in range(4):
            dfmatch = df_phs[(df_phs.fqs >= 17) & (df_phs.pageno == q[1])].copy(deep=True)
            dfmatch = dfmatch[dfmatch.text.str.startswith(q[0])]
            if dfmatch.empty:
                q[0] = q[0].rsplit(" ", 1)[0].strip()
            else:
                return dfmatch.index[0]
    return df_phs.loc[match(start):match(end)]


def get_answer(q):
    q = " ".join([x.strip(punctuation) for x in q.split(" ")])
    q = nlp(q if isinstance(q, unicode) else unicode(q))
    q = u" ".join([(x.lemma_ if x.text not in ["data"] else x.text)
                   for x in q if not (x.is_stop or x.lower_ in ["bot", "help"])])
    q = nlp(q)
    q_set = set(q.text.split())
    def safe_hmean(citem):
        try:
            return hmean((citem.iloc[0], citem.iloc[1]))
        except:
            return 0.
    ntdf = dtls.assign(
        treated=lambda x: x.title.str.apply(unicode).apply(lambda x: nlp(u" ".join([(w.lemma_ if w.text not in ["data"] else w.text) for w in nlp(x.strip()) if not w.is_stop])))).assign(sc_sim=lambda x: x.treated.apply(lambda y: q.similarity(y))).assign(sc_eq=lambda x: x.treated.apply(lambda x: x.text).str.split(" ").apply(lambda y: float(len(q_set.intersection(y))) / max(len(q_set), len(y)))).copy(deep=True)
    ntdf = ntdf.assign(sc_total=lambda x: x[["sc_sim", "sc_eq"]].apply(safe_hmean, axis=1))
    ntdf = ntdf.sort_values(["sc_total", "sc_eq"], ascending=[0, 0]).drop_duplicates(subset="title").iloc[:10][["title", "sc_total", "pageno"]]
    b = inxter(ntdf.iloc[0])
    rpages = [(groupname, groupdf) for (groupname, groupdf) in get_phs_pg(*b).groupby("pageno")]
    return rpages
