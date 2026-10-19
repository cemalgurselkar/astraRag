# AstraRAG Mimari ve Kod Rehberi

Bu belge projeye yeniden girildiğinde kaynak kodu baştan okumadan güncel yapıyı anlamak için tutulur. Kod tabanının 18 Ekim 2026 tarihindeki sadeleştirilmiş tasarımını açıklar.

## Tasarım ilkesi

AstraRAG bilinçli olarak düz bir yapı kullanır:

- Veri sözleşmeleri Pydantic modelleridir.
- Model, client veya indeks durumu taşıyan bileşenler sınıftır.
- Stateless ve deterministik dönüşümler fonksiyondur.
- Birbirinin yerine geçen retrieval stratejileri tek bir `Retriever` protokolünü karşılar.
- Üretimde kullanılmayan deneyler `experiments/` altında tutulur.
- Alt paket ve re-export katmanı yalnız gerçek bir sınır olduğunda kullanılır.

Amaç en az dosya değil, bir sorgunun akışını birkaç dosya açarak baştan sona takip edebilmektir.

## Sistem akışı

```text
İndeksleme
PDF -> clean_text -> parse_pdf -> chunk_document_by_tokens
    -> llama.cpp / all-MiniLM-L6-v2 -> HNSW Qdrant collection
    -> atomic alias replacement

Uygulama başlangıcı
Qdrant'tan chunk'ları oku -> BM25Index kur
    -> dense / hybrid / reranked rotalarını oluştur

Sorgu
query -> profile_query -> choose_route
    -> Retriever.retrieve -> build_context
    -> GeminiGenerator.generate -> PipelineResult
```

Rotalar:

| Rota | Retriever |
|---|---|
| `cheap` | `BM25Index` |
| `medium` | `HybridRetriever` |

## Güncel dosya ağacı

```text
src/astrarag/
├── __init__.py                 Public `AstraRAG` export'u
├── config.py                   Ortam ve uygulama ayarları
├── models.py                   Bütün Pydantic veri sözleşmeleri
├── documents.py                PDF temizleme, parse ve production chunking
├── embedding.py                llama.cpp HTTP embedding client'ı
├── indexes.py                  HNSW Qdrant ve BM25 indeksleri
├── indexing.py                 Production corpus indeksleme fonksiyonu
├── retrieval.py                Production retrieval stratejileri
├── routing.py                  Query profiling ve rota fonksiyonları
├── context.py                  Production context oluşturma fonksiyonu
├── generation.py               Prompt ve Gemini client
├── pipeline.py                 Query-to-answer orkestrasyonu
├── evaluation.py               Dataset, relevance ve metrik fonksiyonları
├── app.py                      Composition root ve public façade
├── cli.py                      index / serve / evaluate komutları
├── api/
│   ├── app.py                  FastAPI uygulaması ve lifespan
│   └── schemas.py              HTTP modelleri
└── experiments/
    ├── documents.py            Parent-child chunking
    ├── retrieval.py            Parent-child, multi-query ve HyDE
    └── context.py              Clustering ve MMR
```

Ana paket 53 Python dosyası ve 2605 satırdan 22 dosya ve yaklaşık 1764 satıra indirilmiştir. Legacy `data/raw` ve eski `attention` şemasına bağlı altı script kaldırılmıştır.

## Dosya sorumlulukları

### `models.py`

Katmanlar arası veri sözleşmelerini tek yerde tutar:

- Belge: `Page`, `Document`, `Chunk`
- Retrieval: `RetrievalResult`
- Routing: `QueryProfile`, `RetrievalRoute`
- Context: `ContextItem`, `ContextPackage`
- Generation: `GeneratedAnswer`
- Pipeline: `PipelineResult`
- Evaluation: `QueryType`, `Evidence`, `EvaluationQuery`

Pydantic bilerek korunmuştur. Modeller farklı dosyalara dağılmadığı için veri akışını takip etmek kolaylaşmıştır.

### `documents.py`

Üç stateless fonksiyon içerir:

- `clean_text(text)`: PDF artefaktlarını ve whitespace'i normalize eder.
- `parse_pdf(path)`: PyMuPDF ile sayfaları çıkarır ve SHA-256 belge kimliği üretir.
- `chunk_document(document, chunk_size, overlap)`: Deneyler için karakter bazlı chunk üretir.
- `chunk_document_by_tokens(...)`: Production için 254 token / 32 overlap chunk üretir.

Bu işlemler kaynak veya değişebilir durum taşımadıkları için sınıf değildir.

### `embedding.py`

`EmbeddingEncoder`, llama.cpp'nin OpenAI uyumlu `/v1/embeddings` endpoint'ine kalıcı HTTP client ile bağlanır. Dönen matrisin adet, boyut ve sonlu sayı kontrollerini yapar; vektörleri L2 normalize eder. Torch, Transformers ve SentenceTransformers production bağımlılığı değildir.

Nihai embedding modeli şimdilik `all-MiniLM-L6-v2` F16 GGUF'tur. Model ve tokenizer revision/SHA-256 değerleri ayarlarda sabittir ve Qdrant collection metadata'sına yazılır.

### `indexes.py`

- `DenseVectorIndex`: Qdrant collection/client durumunu ve HNSW ANN aramasını yönetir.
- `BM25Index`: Tokenize corpus ve BM25 indeksini bellekte tutar.

Qdrant araması `exact=False` ve açık `hnsw_ef` ile yapılır. Collection oluşturulurken `m`, `ef_construct`, full-scan ve indexing threshold değerleri açıkça verilir. Uygulama başlangıcında vector dimension ve index metadata uyumluluğu doğrulanır.

Eski `BM25Retriever` kaldırılmıştır. `BM25Index` doğrudan `retrieve()` sunduğu için tek satırlık adapter'a ihtiyaç yoktur.

### `retrieval.py`

Production retrieval sınırıdır:

- `Retriever`: `retrieve(query, top_k)` yapısal protokolü.
- `DenseRetriever`: Query embedding + Qdrant araması.
- `HybridRetriever`: Dense ve BM25 sonuçlarını RRF ile birleştirir.

Önceki cross-encoder production yolundan kaldırılmıştır. Mevcut benchmark'ta Hybrid'den daha düşük kalite ve yaklaşık 1.2 saniye ek gecikme ürettiği, ayrıca Torch gerektirdiği için kanıtlanmadan geri eklenmemelidir.

Buradaki sınıflar gerçek strateji veya ağır kaynak sahibidir. Eski `retriever()` metot adı `retrieve()` olarak sadeleştirilmiştir.

### `routing.py`

- `profile_query(query)` query sinyallerini çıkarır.
- `choose_route(profile)` retrieval seviyesini seçer.

Eski `QueryProfiler` ve `RuleBasedRouter` sınıfları durum taşımadıkları için kaldırılmıştır.

### `context.py`

`build_context()` sonuç sırasını korur, exact duplicate metinleri atar, item/karakter bütçesini uygular ve `[SOURCE N]` formatı üretir. Eski tek metotlu `ContextEngine` kaldırılmıştır.

### `generation.py`

System instruction ile `GeminiGenerator` aynı modüldedir. Generator harici client taşıdığı ve kapatılması gerektiği için sınıf olarak kalır.

### `pipeline.py`

`RAGPipeline` şu sırayı uygular:

1. Query doğrulama
2. Profil ve rota
3. Seçilen retriever
4. Context oluşturma
5. Generation
6. Retrieval ve toplam latency ölçümü

Pipeline, üç gerçek stratejinin aynı sözleşmeyle değiştirilebilmesi ve fake retriever ile test edilebilmesi nedeniyle sınıf olarak kalır.

### `app.py`

`AstraRAG` composition root ve public façade'dır. Qdrant, embedding modeli, BM25, retriever'lar, Gemini ve pipeline burada birleştirilir. `close()` hem Gemini client'ı hem Qdrant client'ı kapatır.

`Settings.gemini_api_key` opsiyoneldir; yalnız indeksleme çalıştırmak için anahtar gerekmez. `AstraRAG.start()` generation gerçekten kullanılacağı zaman anahtarı zorunlu kılar.

### `indexing.py`

`index_directory()` dosyaları göreli yol, boyut ve `mtime` ile karşılaştırır; yalnız metadata değiştiğinde SHA-256 içeriğini yeniden hesaplar. Corpus tamamen aynıysa PDF parser ve llama.cpp hiç başlatılmaz. Kısmi değişiklikte eski collection'daki değişmeyen vektörler yeniden kullanılır; yalnız yeni veya içeriği değişmiş PDF'ler parse edilip token sınırlarına göre bölünür ve batch'lerle embed edilir.

Her değişiklik yine yeni bir fiziksel collection'a yazılır; başarıdan sonra `astrarag-corpus` alias'ı atomik olarak yeni collection'a alınır. Silinen belgeler yeni collection'a kopyalanmaz, başarısız build mevcut indeksi bozmaz. İndeks şema/model/chunk ayarları uyumsuzsa güvenli bir tam rebuild yapılır. Beş PDF ve 314 chunk içeren mevcut örnek corpus'ta aynı makinedeki tekrar indeksleme 12,42 saniyeden 1,22 saniyeye düşmüştür; mutlak süre donanıma bağlıdır.

Yerel dosya tabanlı Qdrant geliştirme/test için korunur. Docker ve production kullanımı `QDRANT_URL` ile ayrı Qdrant server'a bağlanır; bu sayede API ve indeksleme farklı process'lerde çalışabilir.

### `evaluation.py`

JSON dataset yükleme, evidence-to-chunk çözümleme, relevance kontrolü, recall, reciprocal rank ve nDCG fonksiyonlarını tek modülde tutar.

### `experiments/`

Production bileşimine bağlı olmayan ama deney kanıtı olarak korunan kod:

- Parent-child chunking ve retrieval
- Gemini multi-query
- HyDE
- Clustered context
- MMR context

Context deneyleri stateless fonksiyon, query transformer ve retriever deneyleri client/strateji taşıdıkları için sınıftır.

## Çalışma zamanı nesne grafiği

```text
AstraRAG
├── QdrantClient
├── EmbeddingEncoder (llama.cpp HTTP)
├── DenseVectorIndex
├── BM25Index
├── DenseRetriever
├── HybridRetriever
├── GeminiGenerator
└── RAGPipeline
    └── {cheap, medium} -> Retriever
```

## Scriptlerin durumu

Korunan scriptler:

| Script | Amaç |
|---|---|
| `benchmark_retrieval.py` | Production Dense, BM25 ve Hybrid kalitesiyle llama.cpp/Qdrant p50-p95 gecikmesini ölçer. |
| `benchmark_router.py` | Sabit stratejiler ile rule-based routing'i karşılaştırır. |
| `benchmark_chunking.py` | Chunk boyutu/overlap ablation'ı. |
| `benchmark_context.py` | Baseline, clustered ve MMR context benchmark'ı. |
| `benchmark_advanced_retrieval.py` | Cached multi-query ve HyDE karşılaştırması. |
| `ingest_parent_child.py` | Deneysel parent-child indeks üretimi. |
| `smoke_app.py` | Public API uçtan uca sorgu. |
| `smoke_advanced_retrieval.py` | Multi-query ve HyDE smoke testi. |
| `test_generation.py` | Gemini generation manuel testi. |

Kaldırılan legacy dosyalar Git geçmişinden geri alınabilir: kök `main.py`, `ingest.py`, `inspect_chunks.py`, `benchmark_dense.py`, `debug_retrieval.py`, `first_eval.py`.

## Test durumu

Güncel durumda:

- Testler `test/white_box/` ve `test/black_box/` olarak ayrılmıştır.
- Varsayılan koşuda white-box ve black-box testler birlikte çalışır; gerçek llama.cpp indeksleme, Gemini ve tam retrieval benchmark'ı açıkça opt-in'dir.
- Ruff lint başarılı.
- Mypy, 22 source dosyasında başarılı.
- Python compile kontrolü başarılı.

Black-box testler CLI'ı ayrı proses üzerinden ve API'yi mümkün olan ortamlarda
gerçek Uvicorn/HTTP sınırı üzerinden çalıştırır. Gerçek embedding indeksleme
`RUN_SLOW_TESTS=1`, Gemini servisi `RUN_LIVE_E2E=1`, retrieval kalite kapısı
`RUN_RETRIEVAL_ACCEPTANCE=1` ile açıkça etkinleştirilir.

Testler ayrıca şunları korur:

- `BM25Index` doğrudan retriever olarak çalışır.
- Hybrid duplicate sonuçları doğru RRF skoru ile birleştirir.
- llama.cpp embedding response boyutu ve normalizasyonu doğrulanır.
- Token chunk'ları model limitini aşmaz ve kaynak karakter offset'lerini korur.
- Qdrant collection HNSW ayarları ve model metadata'sıyla oluşturulur.
- Rebuild eski belge chunk'larını aktif indeksten çıkarır.
- Değişmeyen corpus embedding servisi kapalıyken yeniden indekslenebilir.
- Kısmi rebuild değişmeyen belge vektörlerini koruyup yalnız değişen PDF'yi embed eder.

## Kalan darboğazlar ve teknik borç

### Yüksek öncelik

1. **Production retrieval kalitesi.** Güncel 254/32 token + llama.cpp baseline'ında BM25, sabit Hybrid stratejiden daha başarılıdır. 254/64 ablation'ı BM25 Hit@5'i 0.700'den 0.750'ye çıkarırken chunk sayısını %11 artırmıştır; production değişikliği yapılmadan önce bu maliyet değerlendirilmelidir.
2. **BM25 startup maliyeti.** Her proses bütün Qdrant chunk'larını belleğe alıp BM25'yi yeniden kuruyor.
3. **Gemini dayanıklılığı.** Timeout, retry/backoff ve rate-limit politikası yok.
4. **İndeksleme gözlemlenebilirliği.** Artımlı indeksleme belge/chunk sayılarını bildiriyor; yapılandırılmış süre, hata ve build kimliği logları henüz yok.
5. **Collection temizliği.** Alias değişimi güvenli; önceki fiziksel collection silinemediğinde yalnız uyarı basılıyor. Periyodik orphan cleanup komutu yok.

### Orta öncelik

1. BM25 top-k için bütün corpus'u sıralıyor (`O(N log N)`).
2. Dense ve sparse hybrid kolları ardışık çalışıyor.
3. Context bütçesi token yerine ham karakter sayıyor ve source format maliyetini içermiyor.
4. Router İngilizce substring kurallarına bağlı ve kalite kazancı kanıtlanmış değil.
5. `astrarag evaluate` halen gerçek runner yerine taslak komut.
6. Benchmark scriptlerinde metrik ve kurulum tekrarı bulunuyor.
7. PDF parser OCR, tablo ve çok sütunlu okuma sırası çözmüyor.

## Değişiklik yaparken korunacak kurallar

- Pydantic modelleri katmanlar arası sözleşme olarak kalmalı.
- Her production retriever `retrieve()` ile yüksek skor önce döndürmeli.
- Dış skor alanı yalnız `score` olmalı.
- Ağır client/index sahipleri sınıf ve açık yaşam döngüsü kullanmalı.
- Torch/SentenceTransformers production inference yoluna geri eklenmemeli; embedding llama.cpp servisi üzerinden çalışmalı.
- Aktif Qdrant collection, embedding ve chunk metadata'sıyla uyumlu değilse uygulama başlamamalı.
- Stateless dönüşümler gereksiz sınıfa çevrilmemeli.
- Deneysel teknikler ölçülmeden production composition'a eklenmemeli.
- Context citation formatı prompt içindeki `[SOURCE N]` kuralıyla eşleşmeli.
- Yeni bir dosya veya katman, ancak bağımsız değişim ya da gerçek bir sınır sağlıyorsa eklenmeli.
