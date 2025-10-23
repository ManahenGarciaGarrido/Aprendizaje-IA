# Proyecto 13: Traductor Neuronal Multilingüe

## Descripción
Construye un sistema de traducción automática entre múltiples idiomas.

## Duración estimada: 16-20 horas

## Dataset
- **OPUS-100 Dataset**
- Corpus paralelo de 100 idiomas
- Subset: Inglés ↔ Español ↔ Francés

## Tecnologías
- PyTorch/TensorFlow
- Transformers (mT5, mBART)
- SentencePiece tokenizer

## Pasos a Seguir
1. [ ] Cargar OPUS-100 dataset
2. [ ] Preprocesar pares de traducción
3. [ ] Tokenización con SentencePiece
4. [ ] Fine-tuning de mT5/mBART
5. [ ] Implementar beam search
6. [ ] Evaluar con BLEU score
7. [ ] Detección automática de idioma
8. [ ] Crear API de traducción

## Métricas
- BLEU > 35 para inglés-español
