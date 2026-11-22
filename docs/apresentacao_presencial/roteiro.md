# Road Risk MG — Roteiro de apresentação

**14 slides principais**, com duração-alvo de **12 minutos**.

A página final de referências é material de apoio: não precisa ser apresentada.

Os tempos incluem as transições.

## 01. Abertura

**Intervalo:** 00:00–00:25 · **Duração:** 25 s

O objetivo é prever a presença de
acidentes registrados por semana em trechos das rodovias federais de Minas
Gerais.

Conhecer a conectividade da malha ajuda um modelo
a prever melhor? A comparação é entre modelos tabulares e redes neurais em
grafos, ou GNNs.

## 02. Problema e pergunta

**Intervalo:** 00:25–01:10 · **Duração:** 45 s

A utilidade é priorizar fiscalização e inspeções. Em cada unidade segmento–semana,
o alvo vale um quando
há pelo menos uma ocorrência registrada; múltiplas ocorrências continuam sendo
um positivo. Zero significa ausência de registro sob a cobertura assumida.
A hipótese é que a vizinhança acrescente informação. População, atributos
comuns e divisão temporal são mantidos para comparar os modelos.

## 03. Trabalhos correlatos

**Intervalo:** 01:10–01:55 · **Duração:** 45 s

No Brasil, Araújo descobre regras de associação e Amorim classifica severidade
em hotspots. São tarefas distintas. Internacionalmente, Nippani avalia GNNs,
Zhang explora outras modalidades e Gao modela contagem e incerteza.
Esses estudos motivam a hipótese, mas não garantem o mesmo resultado em MG.
A lacuna é uma comparação temporal controlada no contexto mineiro.

## 04. Dados e curadoria

**Intervalo:** 01:55–02:40 · **Duração:** 45 s

São oito arquivos da PRF, de 2019 a julho de 2026, e a malha histórica
do OSM de dezembro de 2018. Após validar os registros de MG,
67.023 ocorrências seguiram para associação ao segmento elegível mais próximo,
em até 500 metros. Foram aceitas 66.186, ou 98,75%.
São ocorrências, não unidades positivas semanais. Distâncias e exclusões
ficaram registradas para auditoria; 2026 é parcial.

## 05. Variáveis de cada fonte

**Intervalo:** 02:40–03:35 · **Duração:** 55 s

Na PRF, `id` controla
identificação e duplicatas; `uf` filtra MG; `data_inversa` e `horario` posicionam
a ocorrência na semana; `latitude` e `longitude` associam o ponto ao trecho.
`br` e `km` são metadados curados, não entradas dos modelos. Causa, gravidade
e clima da ocorrência também não são usados para prever a própria ocorrência.
No OSM, `highway`, `lanes` e `maxspeed` dão tipo de via, faixas e velocidade.
A geometria dá comprimento e sinuosidade; junções criam adjacência.
Referências e IDs apoiam o recorte federal e a rastreabilidade.
O calendário é gerado pela semana de previsão, conhecida antecipadamente;
não é uma característica pós-acidente extraída da PRF.

## 06. Construção do grafo

**Intervalo:** 03:35–04:30 · **Duração:** 55 s

Na malha, junções são nós e trechos são arestas. No grafo
das GNNs, cada trecho físico vira um nó. A, B e C compartilham uma junção,
por isso se tornam vizinhos entre si. Cruzamentos apenas geométricos não
criam conexão. Pistas paralelas, componentes desconectados e isolados
são preservados. São 10.973 segmentos, 14.636 pares adjacentes e 92 componentes.
As GNNs combinam atributos do próprio trecho com mensagens dos vizinhos.

## 07. Atributos e modelos

**Intervalo:** 04:30–05:15 · **Duração:** 45 s

Valores ausentes recebem tratamento e indicadores. Random Forest, XGBoost
e MLP não recebem adjacência; GraphSAGE e GATv2 recebem.
Históricos de acidentes foram previstos no pipeline, mas ficaram ausentes:
não havia prova de publicação dos registros antes da previsão.
O componente temporal efetivamente usado é principalmente calendário.

## 08. Avaliação temporal

**Intervalo:** 05:15–06:10 · **Duração:** 55 s

Treino de 2019 a 2023, validação em 2024,
teste em 2025 e teste adicional em 2026 parcial. Entram semanas completas
e cobertas. Pré-processamento é ajustado no treino; configurações e limiares
são definidos em 2024 e congelados. Foram cinco sementes e 55 treinamentos
com ablações; redes neurais receberam cinco épocas, sem comprovar convergência.
A base tem 4,3 milhões de unidades, mas só cerca de 1,4% são positivas.
AP avalia a ordenação da classe positiva; acurácia seria pouco informativa.

## 09. Resultado principal

**Intervalo:** 06:10–07:15 · **Duração:** 65 s

AP no eixo horizontal, com valores maiores
indicando melhor desempenho. Verde-petróleo representa 2025; azul-escuro,
2026 parcial.
Médias de cinco sementes; hastes mostram um desvio-padrão. XGBoost lidera
com AP de 16,75% em 2025 e 17,23% em 2026. MLP e Random Forest ficam próximos;
as GNNs ficam atrás. Em 2025, a AP do XGBoost é cerca de 11,6 vezes a prevalência.
AP não é a parcela de acidentes detectados nem a precisão de um único limiar.
O padrão se mantém no teste adicional, sem ajuste; isso não comprova
tendência anual, pois 2026 é parcial e condições externas não foram controladas.

## 10. F1 e valores baixos

**Intervalo:** 07:15–08:30 · **Duração:** 75 s

F1 entre cerca de 18% e 25% nos dois testes.
Random Forest tem o maior F1 médio em 2025; XGBoost, em 2026 por pequena margem.
É F1 da classe com acidente, não uma média entre as classes.
F1 combina precisão e revocação e não recebe crédito por negativos corretos.
A classe é rara: muitos negativos fazem com que mesmo uma pequena fração
de falsos alarmes possa representar muitos falsos positivos.
Os modelos têm precisão de aproximadamente 15% a 21% e revocação de 24% a 38%:
há muitos falsos positivos e muitos positivos não detectados. Essa combinação
explica diretamente o F1 baixo. Desbalanceamento sozinho não obriga F1 baixo.
O limiar foi otimizado para F1 em 2024, não nos testes; AUROC alta não garante
um bom ponto de operação. Falta de tráfego, históricos e orçamento curto
são limitações possíveis, não causas isoladas demonstradas pelo experimento.
Cada F1 foi calculado por semente antes da média; a fórmula aplicada às médias
de precisão e revocação não precisa reproduzir exatamente o F1 médio da tabela.

## 11. Ablações

**Intervalo:** 08:30–09:30 · **Duração:** 60 s

Uma ablação remove parte do modelo. Sem topologia, mensagens dos vizinhos
são retiradas, mas a transformação do próprio nó é mantida.
No GraphSAGE, AP completa de 13,86% contra 13,26%: ganho de 0,60 p.p.
No GATv2, 10,37% contra 13,70%: perda de 3,33 p.p.
Sem atributos viários, ambas caem para perto de 1,5%, próximo da prevalência.
Esse foi o principal sinal nas ablações. Remover calendário melhora um pouco
a AP. São diferenças descritivas, não efeitos causais ou testes de significância.

## 12. Interpretação prática

**Intervalo:** 09:30–10:10 · **Duração:** 40 s

No XGBoost de 2025, precisão de 19,39%
e revocação de 35,74%. Entre aproximadamente cinco unidades sinalizadas,
uma é positiva e quatro são falsos positivos; a maioria dos positivos não
é detectada. É suporte analítico, não uma aplicação operacional validada.
Sem volume de tráfego, também não é risco ajustado à exposição.

## 13. Limitações e reprodução

**Intervalo:** 10:10–11:05 · **Duração:** 55 s

Cobertura assumida; malha estática; ausência de volume de tráfego e históricos;
cinco épocas; mesma geografia. Isso delimita a conclusão, não prova
inferioridade geral das GNNs. O pacote reconstruiu os dados e repetiu
inferências em processo separado, sem rede nem treinamento, com diferença
zero nas métricas. É reprodução local no mesmo ambiente, não replicação externa.

## 14. Conclusão

**Intervalo:** 11:05–12:00 · **Duração:** 55 s

Topologia não garantiu melhor predição; pequeno ganho
no GraphSAGE e perda no GATv2, sem superar XGBoost em AP.
A contribuição é a comparação brasileira auditável, temporalmente separada
e reproduzível, mesmo com resultado negativo. Próximos passos: evidência
de cobertura, tráfego e mudanças da malha, novas configurações.
Os testes atuais já foram observados: uma nova busca exige outro protocolo.

## Arquivos e recompilação

- [slides.pdf](./slides.pdf): apresentação pronta, em formato 16:9.
- [slides.html](./slides.html): fonte editável em HTML/CSS, sem dependências externas.
- [malha-mg.png](./malha-mg.png): versão em cinza e recortada da figura histórica do relatório.
- [roteiro.md](./roteiro.md): este guia de fala.

Abra [slides.html](./slides.html) no Chrome ou Chromium e use **Imprimir > Salvar como PDF**.
Ative gráficos de plano de fundo, desative cabeçalhos e rodapés do navegador
e use escala de **100%**. O CSS de impressão define o formato 16:9, sem margens,
com um slide por página. Confira as 15 páginas na prévia antes de salvar.

O PDF foi exportado com Chromium, preservando os fundos e o tamanho definido
no CSS. O navegador usado para a exportação não foi adicionado ao projeto.
Os gráficos e tabelas reproduzem os resultados do relatório e dos CSVs
arquivados. Se os resultados mudarem, revise também esses valores no fonte.
