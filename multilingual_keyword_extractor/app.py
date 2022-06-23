import gradio as gr
import yake
from yake.highlight import TextHighlighter
import pandas as pd

def yake_func(text,slider,slider2):
  kw_extractor = yake.KeywordExtractor(top=slider)
  keywords = kw_extractor.extract_keywords(text)
  text_highlight = TextHighlighter(max_ngram_size = slider2,highlight_pre = "<strong style='color:#538b01'>", highlight_post= "</strong>")
  th=text_highlight.highlight(text, keywords)
  kw = pd.DataFrame(keywords)
  kw.rename(columns={0:'keyword',1:'score'},inplace =True)
  return kw,th

ex1 ="Japan official says highly radioactive water is leaking from crippled nuclear plant into ocean Japan official says highly radioactive water is leaking from crippled nuclear plant into ocean (AP) TOKYO (AP) — Japan official says highly radioactive water is leaking from crippled nuclear plant into ocean."
ex2 = "¿Cómo empezó la Guerra Civil Española?. El mismo año 1936 se celebraron elecciones generales en España, exactamente el 16 de febrero de 1936. A estas elecciones se presentaron muchos partidos políticos tanto de izquierdas como de derechas. El Frente Popular, la coalición de izquierdas que englobaba tanto al Partido Socialista Obrero Español como al Partido Comunista, Izquierda Republicana y otros tantos, consiguió la mayoría absoluta. Pero, ¿cómo comenzó exactamente la Guerra Civil Española? Tras la victoria del bando de izquierda continuaron una serie de acciones terroristas que pretendían movilizar a la masa contra el gobierno, en el caso de los atentados de los falangistas y grupos de derecha, y para responder a los primeros en el caso de los grupos de izquierdas. Solo en el mes de febrero ya se contabilizaban por centenares los fallecidos en este tipo de acciones contra la situación política, social y económica del país. En los meses sucesivos el panorama social y militar de España fue, de todo, menos tranquilo. Varios altos mandos militares planearon durante meses una posible sublevación frente al gobierno republicano que se haría efectiva el 17 de julio de 1936 y los días sucesivos. Pero, ¿qué hizo que los militares se alzaran justo ese día? El 16 de abril de 1936 uno de los hombres de José Castillo, un instructor de las milicias de la juventud socialista, asesinó a Andrés Sáenz de Heredia, primo del mismísimo José Antonio Primo de Rivera. Como represalia el 12 de julio fue asesinado el propio José Castillo. Este hecho desencadenó la venganza de la izquierda que terminó con la vida del diputado de Renovación Española, José Calvo Sotelo, al mismo día siguiente. Este asesinato del líder de la derecha terminó por decantar la balanza de los indecisos al golpe de estado (entre los que, según Paul Preston, se encontraba el propio Franco) a llevar a cabo una acción que conllevaría un conflicto bélico en nuestro país. Así comenzaría la Guerra Civil Española que duraría hasta el 1 de abril de 1939 con la victoria del bando nacional con el general Francisco Franco a la cabeza. Él mismo tomaría las riendas de España bajo un régimen dictatorial hasta su muerte el 20 de noviembre de 1975."


title = "Multi-lingual Keyword Extractor"
description = "Gradio demo of YAKE!, a system for multi-lingual keyword extraction .Just paste or write your text to get the keywords (lower score means more relevant) and see the higlighted version of your text. "
article = "<div style='text-align: center;'>YAKE! : <a href='https://github.com/LIAAD/yake' target='_blank'>Github Repo</a> | <a href='https://repositorio.inesctec.pt/server/api/core/bitstreams/ef121a01-a0a6-4be8-945d-3324a58fc944/content' target='_blank'>Paper</a></div>"

iface = gr.Interface(yake_func, inputs=[gr.inputs.Textbox(label="Text",lines=10),
            gr.inputs.Slider(minimum=1,maximum=20,step=1,default=3,
                         label="Number of keywords"),
            gr.inputs.Slider(minimum=1,maximum=5,step=1,default=3,
                         label="Max ngram size"),],
             outputs=[ gr.outputs.Dataframe(),gr.outputs.HTML()],
             examples=[[ex1,1,3],[ex2,1,3]], examples_per_page=2, live=False, 
             layout="horizontal", interpretation=None, title=title,
              description=description, article=article)

iface.launch()
