# AstraRAG Mimari ve Kod Rehberi

Bu belge projeye yeniden girildiğinde kaynak kodu baştan okumadan güncel yapıyı anlamak için tutulur. Kod tabanının 17 Ekim 2026 tarihindeki sadeleştirilmiş tasarımını açıklar.

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
PDF -> clean_text -> parse_pdf -> chunk_document
    -> EmbeddingEncoder -> DenseVectorIndex / Qdrant

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
| `expensive` | `RerankedRetriever` |

## Güncel dosya ağacı

```text
src/astrarag/
├── __init__.py                 Public `AstraRAG` export'u
├── config.py                   Ortam ve uygulama ayarları
├── models.py                   Bütün Pydantic veri sözleşmeleri
├── documents.py                PDF temizleme, parse ve production chunking
├── embedding.py                Yerel embedding modeli
├── indexes.py                  Qdrant ve BM25 indeksleri
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
- `chunk_document(document, chunk_size, overlap)`: Sayfa bazlı sabit chunk üretir.

Bu işlemler kaynak veya değişebilir durum taşımadıkları için sınıf değildir.

### `embedding.py`

`EmbeddingEncoder`, yerel SentenceTransformer modeline sahip olduğu için sınıftır. Model yükleme pahalıdır ve örnek yaşam döngüsü boyunca tekrar kullanılır.

### `indexes.py`

- `DenseVectorIndex`: Qdrant collection/client durumunu ve dense aramayı yönetir.
- `BM25Index`: Tokenize corpus ve BM25 indeksini bellekte tutar.

Eski `BM25Retriever` kaldırılmıştır. `BM25Index` doğrudan `retrieve()` sunduğu için tek satırlık adapter'a ihtiyaç yoktur.

### `retrieval.py`

Production retrieval sınırıdır:

- `Retriever`: `retrieve(query, top_k)` yapısal protokolü.
- `DenseRetriever`: Query embedding + Qdrant araması.
- `HybridRetriever`: Dense ve BM25 sonuçlarını RRF ile birleştirir.
- `CrossEncoderReranker`: Yerel cross-encoder modelini taşır.
- `RerankedRetriever`: Base retriever adaylarını reranker'dan geçirir.

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

`index_directory()` production PDF'lerini parse, chunk, embed ve `astrarag-corpus` koleksiyonuna yazar. Parent-child indeks artık varsayılan indekslemenin gizli maliyeti değildir; ayrı deney scriptiyle üretilir. Qdrant client `finally` bloğunda kapatılır.

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
├── EmbeddingEncoder
├── DenseVectorIndex
├── BM25Index
├── DenseRetriever
├── HybridRetriever
├── CrossEncoderReranker
├── RerankedRetriever
├── GeminiGenerator
└── RAGPipeline
    └── {cheap, medium, expensive} -> Retriever
```

## Scriptlerin durumu

Korunan scriptler:

| Script | Amaç |
|---|---|
| `benchmark_retrieval.py` | Production ve parent-child stratejilerini karşılaştırır. |
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

Sadeleştirme sonrası:

- Testler `test/white_box/` ve `test/black_box/` olarak ayrılmıştır.
- Varsayılan koşuda 39 test başarılı, 4 ortam-bağımlı test skip edilir.
- Ruff lint başarılı.
- Mypy, 22 source dosyasında başarılı.
- Python compile kontrolü başarılı.

Black-box testler CLI'ı ayrı proses üzerinden ve API'yi mümkün olan ortamlarda
gerçek Uvicorn/HTTP sınırı üzerinden çalıştırır. Gerçek embedding indeksleme
`RUN_SLOW_TESTS=1`, Gemini servisi `RUN_LIVE_E2E=1`, retrieval kalite kapısı
`RUN_RETRIEVAL_ACCEPTANCE=1` ile açıkça etkinleştirilir.

Yeni retrieval testleri şunları korur:

- `BM25Index` doğrudan retriever olarak çalışır.
- Hybrid duplicate sonuçları doğru RRF skoru ile birleştirir.
- Cross-encoder sonucu public `score` alanını günceller; eski `scores` kusuru geri gelemez.

## Kalan darboğazlar ve teknik borç

### Yüksek öncelik

1. **Cross-encoder eager yükleniyor.** Expensive rota kullanılmasa bile startup süresi ve RAM maliyeti oluşuyor. Lazy loading ve kalite/latency yeniden ölçümü gerekiyor.
2. **BM25 startup maliyeti.** Her proses bütün Qdrant chunk'larını belleğe alıp BM25'yi yeniden kuruyor.
3. **İndeksleme tek büyük batch kullanıyor.** Corpus büyüdüğünde chunk listesi ve embedding matrisi belleği artıracak.
4. **Stale Qdrant kayıtları.** Değişen veya silinen PDF'lerin eski chunk'ları otomatik temizlenmiyor.
5. **Gemini dayanıklılığı.** Timeout, retry/backoff ve rate-limit politikası yok.

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
- Ağır model/client/index sahipleri sınıf ve açık yaşam döngüsü kullanmalı.
- Stateless dönüşümler gereksiz sınıfa çevrilmemeli.
- Deneysel teknikler ölçülmeden production composition'a eklenmemeli.
- Context citation formatı prompt içindeki `[SOURCE N]` kuralıyla eşleşmeli.
- Yeni bir dosya veya katman, ancak bağımsız değişim ya da gerçek bir sınır sağlıyorsa eklenmeli.
