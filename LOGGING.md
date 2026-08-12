PS C:\Users\IDFMA0\Desktop\Projects\multi-agent-rag-platform> python .\inference_agent.py

================================================================================
FLUIDRA MULTI-AGENT TECHNICAL RAG
================================================================================

Agents:
  - Orchestrator / Planner
  - SQL / Metadata Agent
  - Semantic Retrieval Agent
  - Guardrails / Security Agent

Tools:
  - semantic_search
  - sql_query

Type 'exit' to quit.

>>> En que tipo de piscinas se pueden instalar este tipo de bombas?

Planning...


[CACHE MISS] orchestrator_agent
[Rate limit protection] Waiting 60 seconds before LLM call...
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[CHECKPOINT] Loaded cached response for guardrails_agent
[CACHE HIT] guardrails_agent
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
[CHECKPOINT] Loaded cached response for retrieval_agent
[CACHE HIT] retrieval_agent
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(

[CACHE MISS] sql_agent
[Rate limit protection] Waiting 60 seconds before LLM call...
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
Saving checkpoint!
[CHECKPOINT] Saved response for sql_agent
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(

[CACHE MISS] synthesis_agent
[Rate limit protection] Waiting 60 seconds before LLM call...
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
Saving checkpoint!
[CHECKPOINT] Saved response for synthesis_agent
C:\Users\IDFMA0\AppData\Roaming\Python\Python314\site-packages\langchain_google_genai\chat_models.py:3120: UserWarning: Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; the sampling parameter(s) temperature will be ignored.
  request = self._build_request_config(
Saving checkpoint!
[CHECKPOINT] Saved response for orchestrator_agent

ANSWER
--------------------------------------------------------------------------------
PREGUNTA DEL USUARIO:
En que tipo de piscinas se pueden instalar este tipo de bombas?

EVIDENCIA Y CONTEXTO ACUMULADO:
Document: user_manual.pdf
- Las bombas para piscinas están diseñadas para la obtención del prefiltrado y recirculación del agua en piscinas.
- Deben trabajar con aguas limpias y con una temperatura que no exceda los 35 °C.
- Deben montarse e instalarse exclusivamente en piscinas que cumplan con las normas IEC / HD 60364-7-702 y la normativa nacional correspondiente.
- Restricciones de ubicación / zonificación eléctrica alrededor de la piscina (según la norma y figura de la página 97):
  * NO se pueden instalar bajo ningún concepto en la **Zona 0** (interior del vaso de la piscina) ni en la **Zona 1** (perímetro inmediato de seguridad que se extiende hasta 2,0 metros horizontalmente desde el borde de la piscina y 2,5 metros de altura).
  * SÍ se pueden instalar en la **Zona 2** (área situada entre los 2,0 y los 3,5 metros de distancia horizontal desde el borde del vaso, y hasta una altura de 2,5 metros), siempre respetando las normas de seguridad eléctrica (protección diferencial de máximo 30 mA, conexión a tierra, soporte fijo, etc.).
- Emplazamiento físico recomendado:
  * Montaje en posición horizontal.
  * Preferiblemente por debajo del nivel del agua de la piscina o estanque (para mejorar el rendimiento y evitar problemas de autoaspiración), asegurando que esté a salvo de inundaciones y cuente con ventilación seca.

================================================================================

>>> 
