# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lib.lang.detect.CommonWords import CommonWords


class Indonesian(CommonWords):

    # We assume 15% as minimum
    MIN_EN_SENT_INTERSECTION_PCT = 0.15

    def __init__(
            self
    ):
        super().__init__()
        self.raw_words = \
"""
sebagai
saya
nya
bahwa
dia
adalah
untuk
pada
adalah
dengan
mereka
menjadi
di
satu
memiliki
ini
dari
oleh
hot
kata
tapi
apa
beberapa
adalah
itu
anda
atau
memiliki
itu
dari
untuk
dan
sebuah
di
kami
bisa
out
lainnya
yang
yang
melakukan
mereka
waktu
jika
akan
bagaimana
kata
an
masing-masing
memberitahu
tidak
Kumpulan
tiga
ingin
udara
baik
juga
bermain
kecil
end
menempatkan
rumah
baca
tangan
pelabuhan
besar
mantra
tambahkan
bahkan
tanah
di sini
harus
besar
tinggi
seperti
ikuti
tindakan
mengapa
bertanya
laki-laki
perubahan
pergi
cahaya
jenis
off
perlu
rumah
gambar
coba
kami
lagi
hewan
titik
ibu
dunia
dekat
membangun
diri
bumi
ayah
apapun
baru
pekerjaan
bagian
mengambil
mendapatkan
tempat
membuat
hidup
mana
setelah
kembali
sedikit
hanya
putaran
pria
tahun
datang
menunjukkan
setiap
baik
saya
memberikan
kami
di bawah
nama
sangat
melalui
hanya
formulir
kalimat
besar
berpikir
mengatakan
membantu
rendah
baris
berbeda
gilirannya
penyebab
banyak
berarti
sebelum
bergerak
kanan
anak
old
terlalu
sama
dia
semua
ada
ketika
naik
penggunaan
Anda
cara
tentang
banyak
kemudian
mereka
menulis
akan
seperti
jadi
ini
dia
panjang
membuat
hal
lihat
dia
dua
memiliki
melihat
lebih
hari
bisa
pergi
datang
melakukan
jumlah
suara
tidak
sebagian besar
orang
saya
lebih dari
tahu
air
dari
panggilan
pertama
yang
mungkin
turun
sisi
berkunjung
sekarang
cari
kepala
berdiri
sendiri
halaman
harus
negara
ditemukan
jawaban
sekolah
tumbuh
penelitian
masih
belajar
tanaman
penutup
makanan
sun
empat
antara
negara
terus
mata
tidak pernah
lalu
biarkan
pikiran
kota
pohon
lintas
pertanian
keras
start
mungkin
cerita
saw
jauh
laut
menarik
kiri
terlambat
Jumlah
tidak
sementara
pers
close
malam
nyata
hidup
beberapa
utara
book
membawa
mengambil
ilmu
makan
kamar
teman
mulai
ide
ikan
gunung
berhenti
sekali
dasar
mendengar
kuda
cut
yakin
menonton
warna
wajah
kayu
utama
terbuka
tampak
bersama-sama
selanjutnya
putih
anak-anak
mulai
mendapat
berjalan
contoh
kemudahan
kertas
kelompok
selalu
musik
mereka
baik
mark
sering
surat
sampai
mil
sungai
mobil
kaki
perawatan
kedua
cukup
polos
gadis
biasa
muda
siap
atas
pernah
merah
daftar
meskipun
merasa
pembicaraan
burung
segera
tubuh
anjing
keluarga
langsung
berpose
meninggalkan
lagu
mengukur
pintu
produk
hitam
pendek
angka
kelas
angin
pertanyaan
terjadi
lengkap
kapal
daerah
setengah
batu
rangka
api
selatan
masalah
piece
mengatakan
tahu
lulus
karena
atas
seluruh
raja
jalan
inch
kalikan
tidak ada
tentu saja
tinggal
roda
penuh
kekuatan
biru
objek
memutuskan
permukaan
dalam
bulan
pulau
kaki
sistem
sibuk
uji
rekor
perahu
umum
emas
mungkin
pesawat
manfaat
kering
bertanya-tanya
tertawa
ribu
lalu
ran
memeriksa
permainan
bentuk
menyamakan
hot
rindu
membawa
panas
salju
ban
membawa
ya
jauh
mengisi
timur
cat
bahasa
antara
Unit
daya
kota
halus
tertentu
terbang
jatuh
memimpin
menangis
gelap
mesin
catatan
menunggu
rencana
angka
bintang
kotak
nomina
lapangan
sisanya
benar
mampu
pound
dilakukan
kecantikan
berkendara
berdiri
berisi
depan
mengajar
minggu
akhir
memberi
hijau
oh
cepat
mengembangkan
ocean
hangat
gratis
menit
kuat
khusus
pikiran
di belakang
jelas
ekor
menghasilkan
Bahkan
ruang
mendengar
terbaik
jam
lebih baik
benar
selama
ratus
lima
ingat
langkah
awal
terus
barat
tanah
bunga
mencapai
cepat
kata kerja
bernyanyi
mendengarkan
enam
tabel
wisata
kurang
pagi
sepuluh
sederhana
beberapa
vokal
menuju
perang
berbaring
terhadap
pola
lambat
pusat
cinta
orang
uang
melayani
muncul
jalan
peta
hujan
aturan
mengatur
menarik
dingin
pemberitahuan
suara
energi
berburu
kemungkinan
tidur
saudara
telur
naik
sel
percaya
mungkin
memilih
tiba-tiba
menghitung
persegi
alasan
panjang
mewakili
seni
Subjek
wilayah
ukuran
bervariasi
menyelesaikan
berbicara
berat
umum
es
peduli
lingkaran
pasangan
termasuk
membagi
suku kata
merasa
agung
bola
belum
gelombang
menjatuhkan
jantung
am
sekarang
berat
tari
mesin
posisi
lengan
lebar
berlayar
materi
fraksi
hutan
duduk
ras
window
toko
musim panas
kereta
tidur
membuktikan
lone
kaki
latihan
wall
catch
gunung
ingin
langit
papan
sukacita
musim dingin
duduk
ditulis
liar
instrument
terus
kaca
rumput
sapi
pekerjaan
tepi
tanda
kunjungan
masa lalu
lembut
menyenangkan
cerah
gas
cuaca
bulan
juta
menanggung
selesai
senang
berharap
bunga
menutupi
aneh
pergi
perdagangan
melodi
perjalanan
kantor
menerima
baris
mulut
tepat
simbol
mati
setidaknya
masalah
berteriak
kecuali
menulis
benih
nada
bergabung
menyarankan
bersih
istirahat
wanita
yard
naik
buruk
pukulan
minyak
darah
sentuh
tumbuh
persen
mencampur
tim
kawat
biaya
hilang
coklat
memakai
garden
sama
dikirim
pilih
jatuh
cocok
mengalir
adil
Bank
mengumpulkan
menyimpan
kontrol
desimal
telinga
lain
cukup
pecah
kasus
tengah
membunuh
putra
danau
saat
skala
keras
musim semi
mengamati
anak
lurus
konsonan
bangsa
kamus
susu
kecepatan
metode
organ
membayar
usia
bagian
dress
awan
kejutan
tenang
batu
kecil
naik
dingin
desain
miskin
banyak
percobaan
bottom
kunci
besi
single
tongkat
datar
dua puluh
kulit
senyum
lipatan
lubang
melompat
bayi
delapan
desa
bertemu
akar
membeli
meningkatkan
memecahkan
logam
apakah
mendorong
tujuh
ayat
ketiga
wajib
diadakan
rambut
menjelaskan
cook
lantai
baik
hasil
membakar
hill
aman
kucing
abad
pertimbangkan
Jenis
hukum
bit
pantai
copy
frase
diam
tinggi
pasir
tanah
gulungan
suhu
jari
industri
nilai
melawan
kebohongan
mengalahkan
menggairahkan
alam
Tampilan
akal
modal
tidak akan
kursi
bahaya
buah
kaya
tebal
tentara
proses
mengoperasikan
praktek
terpisah
sulit
dokter
silahkan
melindungi
siang
tanaman
yang modern
elemen
hit
mahasiswa
sudut
partai
pasokan
yang
mencari
cincin
karakter
serangga
tertangkap
periode
menunjukkan
radio
berbicara
atom
manusia
sejarah
efek
listrik
mengharapkan
tulang
rel
bayangkan
menyediakan
setuju
demikian
lembut
wanita
kapten
menebak
diperlukan
tajam
sayap
membuat
tetangga
mencuci
kelelawar
agak
kerumunan
jagung
membandingkan
puisi
String
bell
tergantung
daging
menggosok
tube
terkenal
dolar
aliran
takut
penglihatan
tipis
segitiga
planet
terburu-buru
kepala
koloni
jam
tambang
dasi
masukkan
utama
segar
pencarian
mengirim
kuning
pistol
memungkinkan
mencetak
mati
tempat
gurun
setelan
saat ini
angkat
naik
tiba
Master
jalur
induk
shore
divisi
sheet
substansi
mendukung
menghubungkan
posting
menghabiskan
chord
lemak
senang
asli
share
stasiun
ayah
roti
biaya
tepat
bar
Penawaran
ruas
budak
bebek
instant
pasar
gelar
mengisi
cewek
sayang
musuh
membalas
minuman
terjadi
mendukung
pidato
alam
Kisaran
steam
gerak
path
cair
login
berarti
quotient
gigi
shell
leher
oksigen
gula
kematian
cukup
keterampilan
wanita
musim
larutan
magnet
perak
terima
cabang
pertandingan
akhiran
terutama
ara
takut
besar
adik
baja
mendiskusikan
maju
serupa
panduan
pengalaman
Rata
apple
membeli
dipimpin
nada
mantel
massa
kartu
pita
tali
Slip
menang
mimpi
malam
kondisi
pakan
alat
total
dasar
bau
lembah
atau
ganda
kursi
terus
blok
grafik
hat
menjual
keberhasilan
perusahaan
kurangi
peristiwa
tertentu
kesepakatan
berenang
istilah
sebaliknya
istri
sepatu
bahu
penyebaran
mengatur
camp
menciptakan
kapas
Lahir
menentukan
liter
sembilan
truk
noise
tingkat
kesempatan
mengumpulkan
shop
stretch
membuang
bersinar
properti
kolom
molekul
pilih
salah
abu-abu
ulangi
membutuhkan
luas
mempersiapkan
garam
hidung
plural
kemarahan
klaim
benua
"""
        self.process_common_words()
        return

    def get_min_threshold_intersection_pct(
            self
    ):
        return Indonesian.MIN_EN_SENT_INTERSECTION_PCT


if __name__ == '__main__':
    obj = Indonesian()
    print(obj.common_words)
    exit(0)
