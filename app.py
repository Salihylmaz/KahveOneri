from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
import os

app = Flask(__name__)

class KahveOnericiSistemi:
    def __init__(self):
        self.kahveciler = {}
        self.menu_yukle()
    
    def menu_yukle(self):
        """CSV dosyalarından kahveci menülerini yükle"""
        csv_dosyalari = {
            'Starbucks': 'starbucks_menu.csv',
            'Mikel Coffee': 'mikel_menu.csv'
        }
        
        for kahveci_adi, dosya_adi in csv_dosyalari.items():
            try:
                if os.path.exists(dosya_adi):
                    df = pd.read_csv(dosya_adi)
                    # Özellikler sütununu liste haline getir
                    df['ozellikler'] = df['ozellikler'].apply(lambda x: x.split(',') if pd.notna(x) else [])
                    self.kahveciler[kahveci_adi] = df
                    print(f"{kahveci_adi} menüsü yüklendi: {len(df)} ürün")
                else:
                    print(f"Uyarı: {dosya_adi} dosyası bulunamadı")
            except Exception as e:
                print(f"Hata: {kahveci_adi} menüsü yüklenirken hata: {e}")
    
    def kahveci_listesi_al(self):
        """Mevcut kahvecilerin listesini döndür"""
        return list(self.kahveciler.keys())
    
    def kahve_onerisi_yap(self, kahveci_adi, tercihler):
        """Tercihlere göre kahve önerisi yap"""
        if kahveci_adi not in self.kahveciler:
            return None
        
        df = self.kahveciler[kahveci_adi]
        uygun_kahveler = []
        
        # Her kahveyi tercihlerle karşılaştır
        for index, kahve in df.iterrows():
            kahve_ozellikleri = kahve['ozellikler']
            # Tercihlerin kahvenin özelliklerinde olup olmadığını kontrol et
            if any(tercih in kahve_ozellikleri for tercih in tercihler):
                uygun_kahveler.append(kahve.to_dict())
        
        if not uygun_kahveler:
            # Eğer tercihle tam eşleşme yoksa, rastgele bir kahve öner
            uygun_kahveler = df.sample(n=min(3, len(df))).to_dict('records')
        
        # Rastgele bir kahve seç
        secilen_kahve = random.choice(uygun_kahveler)
        return secilen_kahve
    
    def kahveci_menusu_al(self, kahveci_adi):
        """Belirli bir kahvecinin tüm menüsünü al"""
        if kahveci_adi in self.kahveciler:
            return self.kahveciler[kahveci_adi].to_dict('records')
        return []

# Global kahve önerici sistemi
kahve_sistemi = KahveOnericiSistemi()

@app.route('/')
def ana_sayfa():
    # Templates klasörü kontrolü
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        return '''
        <h1>Template bulunamadı!</h1>
        <p>Lütfen aşağıdaki adımları takip edin:</p>
        <ol>
            <li><code>templates</code> klasörü oluşturun</li>
            <li>HTML kodunu <code>templates/index.html</code> olarak kaydedin</li>
            <li>CSV dosyalarını ana klasöre ekleyin</li>
            <li>Sayfayı yenileyin</li>
        </ol>
        <p>Gerekli dosyalar:</p>
        <ul>
            <li>starbucks_menu.csv</li>
            <li>mikel_menu.csv</li>
        </ul>
        '''
    return render_template('index.html')

@app.route('/kahveciler')
def kahveciler():
    """Mevcut kahvecilerin listesini döndür"""
    return jsonify(kahve_sistemi.kahveci_listesi_al())

@app.route('/menu/<kahveci_adi>')
def kahveci_menusu(kahveci_adi):
    """Belirli bir kahvecinin menüsünü döndür"""
    menu = kahve_sistemi.kahveci_menusu_al(kahveci_adi)
    return jsonify(menu)

@app.route('/oneri', methods=['POST'])
def kahve_onerisi():
    try:
        veri = request.get_json()
        kahveci_adi = veri.get('kahveci')
        tercihler = veri.get('tercihler', [])
        
        if not kahveci_adi:
            return jsonify({'hata': 'Lütfen bir kahveci seçin!'})
        
        if not tercihler:
            return jsonify({'hata': 'Lütfen en az bir tercih seçin!'})
        
        if kahveci_adi not in kahve_sistemi.kahveci_listesi_al():
            return jsonify({'hata': 'Seçilen kahveci bulunamadı!'})
        
        # Kahve önerisi yap
        onerilen_kahve = kahve_sistemi.kahve_onerisi_yap(kahveci_adi, tercihler)
        
        if not onerilen_kahve:
            return jsonify({'hata': 'Tercihlerinize uygun kahve bulunamadı!'})
        
        return jsonify({
            'oneri': onerilen_kahve,
            'kahveci': kahveci_adi,
            'tercihler': tercihler
        })
        
    except Exception as e:
        return jsonify({'hata': f'Bir hata oluştu: {str(e)}'})

@app.route('/istatistikler')
def istatistikler():
    """Kahveci ve menü istatistikleri"""
    stats = {}
    for kahveci_adi in kahve_sistemi.kahveci_listesi_al():
        menu = kahve_sistemi.kahveci_menusu_al(kahveci_adi)
        stats[kahveci_adi] = {
            'toplam_urun': len(menu),
            'ortalama_fiyat': round(sum(item['fiyat'] for item in menu) / len(menu), 2) if menu else 0,
            'kategoriler': list(set(item['kategori'] for item in menu))
        }
    return jsonify(stats)

if __name__ == '__main__':
    print("=== Kahve Önerici Sistemi ===")
    print(f"Yüklenen kahveciler: {kahve_sistemi.kahveci_listesi_al()}")
    print("Uygulama başlatılıyor...")
    app.run(debug=True, host='0.0.0.0', port=5000)