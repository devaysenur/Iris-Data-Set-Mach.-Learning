from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, pandas as pd
from pandas.plotting import scatter_matrix
#grafikleri satır aralarında görmek için
iris=load_iris()
print(iris.keys()) #verisetindeki anahtaları çağırıyor
#'DESCR' verş setinin özetidir.
#çıktı dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
print(iris['DESCR'])
#çıktı uzun
#target name tahmin etmek istedşğimiz çiçeğ,n türlerini gösteriyo
print(iris['target_names'])
print(iris['feature_names']) #nitelik isimleri
print(type(iris['data'])) #veri türleri, n boyutlu bir numpy dizisi imiş.
#çıktısı:<class 'numpy.ndarray'>

print(iris['data'].shape) #verinin kaç boyutlu oldğunu smyleyecek bize

print(iris['data'][:5]) #veri setindeki ilk 5 örneklermin veri değerlerini görmek istersek
#ilk 5 çiçeğin 4 niteliğinin değerleri (sepal length, sepal width, petal l, p w)

print(iris['target'])
# 0: setosa, 1:versicolor, 2:virginica türünü temsil eder.

#veri seti modeli değerlendirmek için 2 parçaya bölünür ilk parça makine öğrenmesi modeli için kullanılır: training yani eğitim setidir.
#diğeri ise modelin nasıl çalıştığını söylemek için kullanılır buna da test verisi denir

'''sklearnde veriyi karıştırmak ve ayırmak için train test split diye bir fonksiyon var
önce satırları karıştırır sonra %75'ini eğitim %25 test olarak ayırır.
skitlearnde veri X ile etiketler ise y ile gösterilir. X iki boyutlu dizi yani matristir.
 y ise tek boyutlu dizi yani vektörü gösterir. f(X)=y ortaya çıkar'''

X_egitim, X_test, y_egitim, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
#veriyi karıştırmanın önemi: veri etiketlerle sıralıdır ama son %25ini alırsak hep aynı etikete sahip grubu test grubu yapar diğeriyle ise eğitmiş oluruz.

print(X_egitim.shape) #(112, 4) 2 boyutlu
print(y_egitim.shape) #(112,) tek boyutlu

print(X_test.shape) #(38, 4) 2 boyutlu
print(y_test.shape)#(38,) tek boyutlu

'''modeli kurmadan veriyi incelemek önemli'''

iris_df=pd.DataFrame(X_egitim, columns=iris.feature_names)



#saçılım matrix grafiği:
scatter_matrix(iris_df, c=y_egitim, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=80, alpha=0.8)
'''c argümanı verinin türlere göre renklenmesi için kullanılır colour
grafiğin boyutu figure size, noktaların şekli marker, histogramların
 dikdörtgen genişlikleri için hist_kwds, nokta büyüklüğü size s, 
 noktaların görünümü alpha'''


