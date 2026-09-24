# AstraRAG Python Kod Rehberi

Bu belge, projeyi bir yapay zekâya hızlıca tanıtmak için Python kodlarının kısa ve güncel bir özetidir.

## Projenin amacı

AstraRAG; PDF belgelerini işleyip Qdrant ve BM25 üzerinde arayan, sorgunun yapısına göre farklı retrieval maliyetleri seçen deneysel bir RAG sistemidir. Embedding ve reranking yerel modellerle, cevap üretimi Gemini API ile yapılır. Sistem Python paketi, CLI ve FastAPI servisi olarak kullanılabilir.

## Ana çalışma akışı

```text
PDF -> temizleme -> chunk üretimi -> embedding -> Qdrant/BM25
Sorgu -> profil çıkarma -> rota seçimi
  cheap: BM25
  medium: Dense + BM25 (RRF)
  expensive: Hybrid + Cross-Encoder
-> context oluşturma -> Gemini ile kaynaklı cevap
```

## Kök dosya

- `main.py`: Tek bir PDF üzerinde parse, sabit boyutlu chunk, embedding, Qdrant'a yazma ve dense retrieval işlemlerini uçtan uca gösteren eski/demo scriptidir.

## Ana paket: `src/astrarag`

- `__init__.py`: Paketin dışarı açılan ana sınıfı `AstraRAG`'ı export eder.
- `config.py`: `.env` ve ortam değişkenlerinden Gemini, model yolları, Qdrant, retrieval ve context ayarlarını okuyan `Settings` sınıfıdır.
- `app.py`: Tüm bileşenleri kuran ana façade'dır. Qdrant'taki chunk'ları yükler; BM25, dense, hybrid ve reranked retriever'ları oluşturur; sorguyu adaptive pipeline'a gönderir.
- `cli.py`: `astrarag index`, `astrarag serve` ve taslak durumdaki `astrarag evaluate` komutlarını sağlar.

### API: `src/astrarag/api`

- `api/app.py`: FastAPI uygulamasını, uygulama yaşam döngüsünü, `GET /health` ve `POST /query` endpoint'lerini tanımlar.
- `api/schemas.py`: HTTP istek/yanıt modellerini (`QueryRequest`, `QueryResponse`, kaynak ve health modelleri) tanımlar.
- `api/__init__.py`: API alt paketini işaretleyen boş modüldür.

### Veri modelleri: `src/astrarag/schemas`

- `schemas/document.py`: Sayfa, belge ve chunk modelleri (`Page`, `Document`, `Chunk`).
- `schemas/retrieval.py`: Bir arama sonucunun metin, skor, sayfa ve metadata alanlarını taşıyan `RetrievalResult`.
- `schemas/routing.py`: `cheap/medium/expensive` rotaları ve sorgu özelliklerini tutan `QueryProfile`.
- `schemas/context.py`: LLM'e verilecek tekil kaynakları ve birleştirilmiş context paketini tanımlar.
- `schemas/generation.py`: Üretilen cevap, model adı ve kullanılan kaynak kimliklerini tutar.
- `schemas/pipeline.py`: Retrieval, context, generation, rota ve süreleri tek sonuçta birleştiren `PipelineResult`.
- `schemas/evaluation.py`: Değerlendirme sorgu türleri, kanıt parçaları ve `EvaluationQuery` modeli.
- `schemas/__init__.py`: Yukarıdaki modelleri paket seviyesinden export eder.

### Belge işleme: `src/astrarag/ingestion`

- `ingestion/cleaner.py`: PDF kaynaklı bozuk karakterleri, satır sonlarını, hece bölünmelerini ve fazla boşlukları temizler.
- `ingestion/parser.py`: PyMuPDF ile metin tabanlı PDF'leri okur; dosya içeriğinden SHA-256 belge kimliği üretir ve sayfaları `Document` modeline çevirir.
- `ingestion/chunker.py`: Her sayfayı karakter tabanlı, örtüşmeli sabit boyutlu chunk'lara böler.
- `ingestion/paren_child_chunker.py`: Küçük parçadan arama, büyük parçadan context verme yaklaşımı için parent ve child chunk'ları üretir. Dosya adındaki `paren` yazımı mevcut kodun adıdır.
- `ingestion/indexer.py`: Bir dizindeki PDF'leri topluca parse eder; normal ve parent-child chunk'ları embed ederek iki Qdrant koleksiyonuna yazar.
- `ingestion/__init__.py`: Parser, cleaner, chunker ve indexer sınıflarını export eder.

### Embedding ve indeksler

- `embedding/encoder.py`: Yerel SentenceTransformer modelini yükler, normalize edilmiş doküman/sorgu embedding'leri üretir.
- `embedding/__init__.py`: `EmbeddingEncoder` export'unu sağlar.
- `index/vector_store.py`: Qdrant cosine koleksiyonunu yönetir; chunk/embedding ekler, vektör araması yapar ve kayıtlı chunk'ları geri okur.
- `index/sparse_index.py`: Basit regex tokenization kullanan bellek içi BM25 indeksini kurar ve arar.
- `index/__init__.py`: Dense ve BM25 indekslerini export eder.

### Retrieval ve reranking

- `retrieval/dense.py`: Sorguyu embed edip Qdrant dense aramasını çalıştırır.
- `retrieval/sparse.py`: BM25 indeksini standart retriever arayüzüne uyarlar.
- `retrieval/hybrid.py`: Dense ve BM25 sonuçlarını Reciprocal Rank Fusion (RRF) ile birleştirir.
- `retrieval/reranked.py`: Bir base retriever'dan geniş aday listesi alıp cross-encoder ile yeniden sıralar.
- `retrieval/parent_child.py`: Child chunk'larda arar, tekrarları kaldırarak ilişkili büyük parent metinlerini döndürür.
- `retrieval/__init__.py`: Retriever sınıflarını paket seviyesinden export eder.
- `reranking/cross_encoder.py`: Yerel CrossEncoder ile sorgu–aday çiftlerini puanlayıp yeniden sıralar.
- `reranking/__init__.py`: `CrossEncoderReRanker` export'unu sağlar.

### Adaptive routing ve pipeline

- `routing/profiler.py`: Sorgunun kelime sayısı, sayı, karşılaştırma, multi-hop ve tırnaklı tam ifade sinyallerini çıkarır.
- `routing/router.py`: Karşılaştırma/multi-hop sorgularını `expensive`; sayı, tam ifade veya uzun sorguları `medium`; diğerlerini `cheap` rotasına yollar.
- `routing/__init__.py`: Profiler ve router'ı export eder.
- `pipeline/adaptive.py`: Profil -> rota -> retrieval -> context -> generation zincirini çalıştırır ve retrieval/toplam gecikmeyi ölçer.
- `pipeline/__init__.py`: `AdaptiveRAGPipeline` export'unu sağlar.

### Context ve cevap üretimi

- `context/engine.py`: Retrieval sonuçlarını metne göre tekilleştirir, öğe/karakter limitine uyar ve `[SOURCE N]` formatında LLM context'i hazırlar.
- `context/__init__.py`: `ContextEngine` export'unu sağlar.
- `generation/prompt.py`: Yalnızca verilen bağlama dayanma ve kaynak gösterme kurallarını içeren sistem talimatını/prompt'u üretir.
- `generation/gemini.py`: Gemini istemcisini çağırır; context yoksa yetersiz kanıt cevabı döndürür, varsa kaynak kimlikleriyle `GeneratedAnswer` üretir.
- `generation/__init__.py`: `GeminiGenerator` export'unu sağlar.

### Değerlendirme: `src/astrarag/evaluation`

- `evaluation/dataset.py`: JSON değerlendirme verisini okuyup Pydantic ile `EvaluationQuery` listesine doğrular.
- `evaluation/ground_truth.py`: Kanıt metnini normalize ederek ilgili chunk'ları bulur ve retrieval sonucunun ilgili olup olmadığını kontrol eder.
- `evaluation/retrieval.py`: Recall@K, reciprocal rank ve nDCG@K metriklerini hesaplar.
- `evaluation/__init__.py`: Veri yükleme, ground-truth ve metrik fonksiyonlarını export eder.

## Yardımcı scriptler: `scripts`

- `scripts/ingest.py`: `data/raw` altındaki PDF'leri sabit chunk'larla eski/usul dense koleksiyona indeksler.
- `scripts/ingest_parent_child.py`: Parent-child child chunk'larını ayrı Qdrant koleksiyonuna indeksler.
- `scripts/benchmark_retrieval.py`: Dense, BM25, hybrid, cross-encoder ve parent-child stratejilerini Hit@K, MRR, nDCG ve latency ile karşılaştırır.
- `scripts/benchmark_router.py`: Sabit retrieval stratejilerini adaptive router ile karşılaştırır ve rota dağılımını raporlar.
- `scripts/benchmark_dense.py`: Tek dense retrieval baseline'ını eski attention değerlendirme verisi üzerinde ölçer.
- `scripts/debug_retrieval.py`: Seçili attention sorgularının ilk 10 sonucunu ayrıntılı yazdırır.
- `scripts/first_eval.py`: Attention PDF için arama ifadelerinden eski formatta ilk değerlendirme JSON'unu üretir.
- `scripts/inspect_chunks.py`: Attention PDF'nin üretilen chunk'larını terminalde incelemek için yazdırır.
- `scripts/test_generation.py`: Yapay bir retrieval sonucu ile Gemini cevap üretimini manuel olarak dener.
- `scripts/smoke_app.py`: `AstraRAG` sınıfıyla gerçek bir uçtan uca sorgu smoke testi yapar.

## Testler: `test`

- `test/test_parse.py`: PDF metin çıkarımını test eder.
- `test/test_chunker.py`: Sabit chunk bölme, overlap ve sayfa numarası davranışını test eder.
- `test/unit/test_cleaner.py`: PDF artefaktı, hece bölünmesi, satır sonu ve boş metin temizliğini test eder.
- `test/unit/test_parent_child_chunker.py`: Parent-child ilişkisinin metadata içinde doğru kurulmasını test eder.
- `test/unit/test_routing.py`: Sorgu profili ve cheap/medium/expensive rota kurallarını test eder.
- `test/unit/test_context_engine.py`: Context üretimi, deduplication ve öğe/karakter limitlerini test eder.
- `test/unit/test_adaptive_pipeline.py`: Sahte retriever/generator ile rota seçimi, context ve latency kaydını test eder.

## Önemli mevcut notlar

- Güncel ana kullanım `src/astrarag`, `data/documents`, `astrarag-corpus` ve `data/eval/retrieval_v1.json` etrafındadır; `main.py` ile bazı eski scriptler `data/raw`, `attention` koleksiyonu veya eski evaluation formatını kullanır.
- Güncel `EvaluationQuery` modeli `evidence` alanı bekler. `benchmark_dense.py`, `debug_retrieval.py` ve `first_eval.py` ise artık modelde bulunmayan `relevant_chunk_ids` tabanlı eski formatla uyumlu yazılmıştır.
- `cli.py` içindeki `evaluate` komutu henüz gerçek değerlendirme çalıştırmaz, yalnızca dosya yolunu yazdırır.
- `cross_encoder.py` yeniden sıralama sonucunda `score` yerine `scores` alanını güncelliyor; `RetrievalResult` alanı `score` olduğu için bu bölüm kontrol edilmelidir.
- Yerel embedding ve reranker model klasörleri ile `.env` içinde `GEMINI_API_KEY` çalışma ön koşuludur. Qdrant dosya tabanlı olarak `data/qdrant` altında tutulur.
