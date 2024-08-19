
Son yıllarda bitki tür tanıma, yaprak, gövde, çiçek vb. çeşitli bitki parçalarının şekil, geometri
ve dokularına dayalı olarak yapılmaktadır. Çiçek tabanlı bitki tür tanımlama sistemleri yaygın
olarak kullanılmaktadır. Modern arama motorları, bir çiçek içeren bir sorgu görüntüsünü görsel
olarak aramak için yöntemler sağlarken, dünya çapında milyonlarca çiçek türü arasındaki sınıf
içi çeşitlilik nedeniyle sağlamlıktan yoksundur. Bu nedenle önerilen bu araştırma çalışmasında,
çiçek türlerini yüksek doğrulukla tanımak için Evrişimli Sinir Ağları (CNN) kullanan bir Derin
öğrenme yaklaşımı kullanılmıştır.


## Geliştirilen Yöntem

Derin Öğrenme araştırmasında, CNN'ler, Görüntü Sınıflandırma ve Nesne Tanıma içeren
Bilgisayarla Görme uygulamaları için özel olarak uygulanır. Çiçek türlerini tanıma hem Nesne
Tanıma hem de Görüntü Sınıflandırmanın bir kombinasyonudur, çünkü sistemin, görüntüdeki
bir çiçeği algılamanın yanı sıra hangi türe ait olduğunu da tanıması gerekir. Çiçek türlerini
tanımak için, akıllı bir sistemin daha geniş görüntü setleriyle eğitilmesi gerekir, böylece çiçek
türlerini öğrenmiş olduğu kalıplardan tahmin edebilir. Bu yaklaşım, görünmeyen bir
görüntünün etiketini tahmin etmek için etiketlere sahip, harici bir görüntü veri seti gerektiren


“Denetimli Öğrenme” olarak adlandırılır. Bu çalışma, çiçek türlerini gerçek zamanlı olarak
verimli bir şekilde tanımak için akıllı algoritma olarak Transfer Learning ile Evrişimli Sinir
Ağlarını (CNN) kullanır. Geleneksel Yapay Sinir Ağı (ANN) ve CNN arasındaki en büyük fark,
bir CNN'nin yalnızca son katmanının tamamen bağlı olması, ANN'de ise her nöronun Şekil 1 'de
gösterildiği gibi diğer nöronlara bağlı olmasıdır. ANN’ ler görüntüler için uygun değildir çünkü
bu ağlar, görüntülerin boyutu nedeniyle kolayca aşırı sığdırmaya yol açar. [32x32x3]
boyutundaki bir görüntü ANN’ e geçirilecekse, 32x32x3 = 3072 satırlık bir vektöre
düzleştirilmesi gerekir. Bu nedenle, ANN’ in bu girdi vektörünü alabilmesi için ilk katmanında
3072 ağırlığa sahip olması gerekir. Daha büyük görüntüler için, örneğin [300x300x3], işlenmesi
için daha güçlü bir işlemci gerektiren karmaşık bir vektör (270.000 ağırlık) ile sonuçlanır.


CNN'ler, bir girdi görüntüsünü alan, matematiksel bir işlem gerçekleştiren (ReLU, tanh gibi
doğrusal olmayan aktivasyon işlevi) ve çıktıdaki sınıf veya etiket olasılıklarını tahmin eden bir
katman yığınından oluşur. CNN'ler, standart el yapımı özellik çıkarma yöntemlerini kullanmak
yerine, giriş görüntüsünün ham piksel yoğunluğunu düzleştirilmiş bir vektör olarak alır.
Örneğin, [30x30] renkli bir görüntü, CNN'nin giriş katmanına 3 boyutlu bir matris olarak
geçirilecektir. CNN, “öğrenilebilir” filtrelere sahip farklı katmanları kullanarak görüntüde
bulunan karmaşık özellikleri otomatik olarak öğrenir ve giriş görüntüsünün sınıf veya etiket
olasılıklarını tahmin etmek için bu filtrelerin sonuçlarını birleştirir. Bir ANN’ den farklı olarak,
bir CNN katmanındaki nöronlar, diğer tüm nöronlara bağlı değildir, ancak önceki katmandaki
sadece küçük bir nöron bölgesine bağlıdır. İlk katman, görüntüdeki köşeler ve kenarlar gibi en
düşük seviyeli özellikleri algılayabilir. Sonraki katmanlar, şekiller ve dokular gibi orta seviye
özellikleri algılayabilir ve son olarak, bitki veya çiçeğin yapısı gibi daha yüksek seviyeli
özellikler, ağdaki daha yüksek katmanlar tarafından algılanacaktır. Bir görüntüdeki daha düşük
seviyeli özelliklerden daha yüksek seviyeli özelliklere geçmenin bu benzersiz tekniği, CNN'leri
birçok uygulamada en kullanışlı yapan şeydir.

Bir Evrişimsel Sinir Ağı (CNN), aşağıdaki gibi üç tür katmana sahiptir:

Evrişim Katmanı (CONV)

Havuzlama Katmanı (POOL)

Tam Bağlantılı Katman (FC)

**Evrişim Katmanı (CONV)**


En çok kullanılan katmandır. Herhangi bir CNN mimarisinde önemli bir katmandır, çünkü bu,
CNN'nin giriş görüntüsünden özellikleri öğrenmek için filtreler uyguladığı katmandır. Bu
katman filtrelerden ve özellik haritalarından oluşur. CONV katmanı, boyut olarak küçük olan
M filtreleri içerir (örneğin [3x3] veya [5x5]). Bu filtreler, tüm görüntüden geçmek yerine, belirli
bir uzamsal konumdaki özellikleri öğrendiği giriş hacmi ile kıvrılır. Her öğrenilebilir filtre için,
filtre ağın genişliği ve yüksekliği boyunca kayarken, girdileri ve girdisi ile nokta ürünlerini
hesaplarken 2 boyutlu bir özellik haritası oluşturulur. Giriş hacmine tüm M filtreleri
uygulandığında, ilgili tüm 2 boyutlu özellik haritaları, nihai çıktı hacmini üretmek için
birleştirilir. Bu çıktı hacmi, girdi görüntüsünde yalnızca belirli bir uzaysal bölgeye bakan
filtrelerden girişlere sahiptir. Örneğin, 64x64x12'lik bir çıktı hacmi elde etmek için sırasıyla
64x64x3 ve 12 boyut ve filtre sayısına sahip giriş görüntüsü kullanılır. Bu filtreler, giriş
görüntüsünde bulunan kenarları veya köşeleri öğrenmiş olacak ve ancak bu benzer kenarları ve
köşeleri tekrar gördüklerinde etkinleşecektir.

**Havuzlama Katmanı (POOL)**

Bu katman, gelen hacmi uzamsal boyutlar boyunca altörneklediği veya sıkıştırdığı ağda bir ara
katman olarak kullanılır. Örneğin, giriş hacmi [64x64x12] ise, alt örneklenmiş hacmi
[32x32x12] olur. Böylece, ağdaki aşırı sığdırma ve hesaplamaları azaltmak için önceki
katmanın farklı filtrelerden elde edilen özellik haritalarını altörnekler.

**Tam Bağlantılı Katman (FC)**

ANN’ lerde olduğu gibi, bir CNN'deki FC katmanı, önceki katmandaki nöronlara tamamen
bağlı nöronlara sahiptir. Bu FC katmanı genellikle çok sınıflı sınıflandırma problemleri için
aktivasyon fonksiyonu olarak “SOFTMAX” olan bir CNN'nin son katmanı olarak tutulur. FC
katmanı, giriş görüntüsünün son sınıfını veya etiketini tahmin etmekten sorumludur. Bu
nedenle, [1x1xN] çıktı boyutuna sahiptir; burada N, sınıflandırma için düşünülen sınıfların veya
etiketlerin sayısını belirtir.

### Veriseti

Şu anda dünyada on binlerce çeşit çiçek kategorisi bulunmaktadır. Bu kadar çok çiçek çeşidini
sınıflandırmak için standart bir çiçek veri seti seçilmelidir. kaggle.com ‘dan alınan ‘Flowers
Recognition’ veriseti 4242 çiçek görüntüsü içerir. Örnek görüntüler, Flickr verilerine, Google
görsellerine, Yandex görsellerine dayanmaktadır. Görüntüler; papatya, lale, gül, ayçiçeği,
karahindiba olmak üzere 5 sınıfa ayrılır. Her sınıf için yaklaşık 800 fotoğraf vardır. Fotoğraflar
yüksek çözünürlükte değil, yaklaşık 320×240 pikseldir. Fotoğraflar tek bir boyuta indirgenmez,
farklı oranlardadır.

### Tensorflow

TensorFlow, Google'ın DistBelief tabanlı ikinci nesil yapay zeka öğrenme sistemidir.
TensorFlow, sayısal hesaplama için veri akışı grafiklerini kullanan açık kaynaklı bir yazılım
kitaplığıdır. TensorFlow, orijinal olarak Google Brain Group’ tan (Google Machine
Intelligence Research Institute’ ye bağlı) araştırmacılar ve mühendisler tarafından makine
öğrenimi ve derin sinir ağı araştırması için geliştirilmiştir, ancak çok yönlülüğü, diğer bilgi
işlem alanlarında da yaygın olarak kullanılmasına olanak tanır. Bir akıllı telefondan binlerce
veri merkezi sunucusuna kadar çeşitli iyileştirmeler üzerinde çalışabilir. TensorFlow tamamen
açık kaynaklıdır ve herkes tarafından kullanılabilir.

### NumPy

NumPy, Python programlama diline ait çok boyutlu dizilerle ve matrislerle çalışmaya yardım
eden ileri düzey matematiksel işlemler yapılabilen bir kütüphanedir. Günümüzde özellikle veri
bilimi üzerine çalışanlar başta olmak üzere Numpy, Python programlayanlar tarafından çok sık
kullanılan bir kütüphanedir.

### Matplotlib

Matplotlib, figürlerin görselleştirilmesinde ve analizine yardımcı olan en popüler 2 Boyutlu bir
çizim kütüphanesidir. Görselleştirme bilgiyi analiz etmenin ve ondan belirli sonuçlar
çıkarmanın en kolay yoludur. Yapılan bu görselleştirmeler karmaşık bir sorunu kolayca
anlamaya yardımcı olur. Görselleştirmeler verideki ilişkileri, yapıları ve aykırı değerleri tespit
etmeye destek sağlar. Aynı zamanda EDA (Exploratory Data Analysis) ve Makine Öğrenmesi
gibi süreçlere veriyi hazırlar.

### Scikit-learn

Scikit-learn, veri bilimi ve machine learning için en yaygın kullanılan Python paketlerinden
biridir. Birçok işlemi gerçekleştirmeyi ve çeşitli algoritmaları sağlar. Scikit-learn ayrıca
sınıfları, yöntemleri ve işlevleri ile kullanılan algoritmaların arka planıyla ilgili belgeler sunar.

### Transfer Learning

Transfer öğrenmesi, bir problemi çözerken elde edilen bilgiyi saklamak ve daha sonra farklı
ama ilgili bir probleme uygulamak üzerine odaklanan makine öğreniminde bir araştırma
problemidir. Örneğin, otomobilleri tanımayı öğrenirken kazanılan bilgi, kamyonları tanımaya
çalışırken uygulanabilir. Bu alandaki araştırmalar, iki alan arasındaki resmi bağlar sınırlı
olmakla birlikte, öğrenmenin aktarımı üzerine psikolojik literatürün uzun tarihi ile bazı ilişkileri
taşımaktadır. Transfer öğrenimi, bir görev için eğitilmiş bir modelin, ilgili ikinci bir görevde
yeniden tasarlandığı bir makine öğrenmesi tekniğidir. Öğrenme ve alan adı uyarlamasını
aktarma, bir ortamda öğrenilenin başka bir ortamda genelleşmeyi iyileştirmek için
kullanılmasıyla ilgilidir. Transfer öğrenimi, ikinci görevi modellerken hızlı ilerleme veya
gelişmiş performans sağlayan bir optimizasyondur. Öğrenmeyi transfer etmek, yeni bir görevde
öğrenmenin, önceden öğrenilmiş olan ilgili bir görevden bilginin aktarılması yoluyla
iyileştirilmesidir. Transfer öğrenmesi, çok görevli öğrenme (multi-tasking learning) ve kavram
sapması gibi problemlerle ilgilidir ve yalnızca derin öğrenme için bir çalışma alanı değildir.
Bununla birlikte, derin öğrenme modellerini veya derin öğrenme modellerinin eğitildiği büyük
ve zorlu veri kümelerini eğitmek için gerekli olan muazzam kaynaklar göz önünde
bulundurulduğunda, transfer öğrenimi derin öğrenmede popülerdir.

