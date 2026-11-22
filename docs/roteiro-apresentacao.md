# Roteiro de apresentação — Road Risk MG

**Duração alvo:** 5 minutos | **Apresentadores:** João, Giovanni e Lion
**Orientação:** leiam as falas em ritmo natural, por volta de 120 a 130 palavras por minuto. As indicações entre colchetes são visuais e não devem ser lidas.

## João — 0:00 a 1:40

[Vídeo ao fundo: nome do projeto, mapa de Minas Gerais e visão geral das pastas.]

Olá! Nós somos João, Giovanni e Lion. Neste vídeo, vamos percorrer a lógica do Road Risk MG: da entrada dos dados até a comparação dos modelos. A pergunta central é: usando dados públicos, conseguimos estimar em quais segmentos das rodovias federais de Minas Gerais foi registrado pelo menos um acidente em determinada semana? Este é um estudo experimental, não um sistema de alerta em operação nem uma medida de risco ajustada pelo volume de veículos.

O código principal fica em `codigo/src/road_accidents`. As configurações em `codigo/configs/` definem fontes, períodos e parâmetros. O arquivo `codigo/dvc.yaml` organiza três etapas: preparar os dados, executar os experimentos e gerar relatórios. A interface de linha de comando, chamada `road-risk`, aciona essas operações. Em vez de concentrar tudo em um único programa, o projeto separa ingestão, atributos, grafos, modelos, avaliação e relatórios em módulos próprios. A pasta `tests/` verifica contratos e integração; `resultados/` reúne tabelas e gráficos, e o relatório em `docs/trab_final.tex` registra método, achados e limitações.

A preparação começa com registros da PRF de 2019 a julho de 2026 e uma malha do OpenStreetMap baseada em uma fotografia histórica de 2018. O código valida o formato e a qualidade dos registros e guarda manifestos e hashes para identificar as fontes usadas.

## Giovanni — 1:40 a 3:20

[Vídeo ao fundo: arquivos da PRF, malha viária e animação de pontos sendo associados a trechos.]

Um cuidado importante: a cobertura temporal foi autorizada como hipótese do projeto, mas não foi comprovada por documentação oficial. Portanto, ausência de registro não prova que nenhum acidente aconteceu.

Na etapa espacial, cada segmento físico da rodovia vira um nó do grafo. Dois nós são ligados quando seus trechos compartilham uma junção da malha. Cada ocorrência válida é associada ao segmento elegível mais próximo, desde que esteja a até 500 metros. Distâncias, rejeições e empates ficam registrados para auditoria.

Depois, o sistema monta uma observação para cada segmento e semana completa. O alvo vale um quando há pelo menos um acidente elegível naquela combinação; caso contrário, vale zero, condicionado à hipótese de cobertura. São 10.973 segmentos e mais de 4,3 milhões de pares segmento-semana, com apenas cerca de 1,4% de observações positivas. O tempo é dividido sem misturar períodos: treino de 2019 a 2023, validação em 2024, teste em 2025 e teste parcial de janeiro a julho de 2026. Os atributos incluem características da via e do calendário. Histórico de acidentes só poderia ser usado se sua disponibilidade antes da previsão estivesse documentada; como isso não foi comprovado, esses campos ficam ausentes. O pré-processador também é ajustado somente nos anos de treino.

## Lion — 3:20 a 5:00

[Vídeo ao fundo: nomes dos modelos, gráficos de avaliação e arquivos de reprodução.]

Com os dados preparados, o projeto compara cinco modelos: Random Forest, XGBoost, MLP, GraphSAGE e GATv2. Os três primeiros usam atributos tabulares; GraphSAGE e GATv2 também recebem as conexões entre segmentos. As configurações ficam em arquivos YAML. Cada modelo é executado com cinco sementes. A validação escolhe o limiar de decisão, que fica congelado antes dos testes. Como os acidentes são raros, a métrica principal é a Average Precision, acompanhada de AUROC, precisão, revocação e F1.

Nos resultados registrados, o XGBoost obteve a maior Average Precision: 16,75% em 2025 e 17,23% no teste parcial de 2026. Os modelos tabulares superaram as redes de grafos. A topologia ajudou pouco o GraphSAGE e não ajudou o GATv2 neste protocolo; portanto, conectar os trechos não garante, por si só, uma previsão melhor.

Por fim, manifestos e hashes identificam código, dados, ambiente e modelos. O experimento é congelado antes da avaliação; os módulos de relatório e reprodução geram tabelas e gráficos e permitem verificar o pacote sem retreinar os modelos. Ainda há limites importantes: cobertura assumida, malha viária estática de 2018, ausência de dados de tráfego e nenhuma avaliação em outros estados. A contribuição principal é um fluxo comparável, auditável e reproduzível — não uma taxa definitiva de risco rodoviário. Obrigado!
