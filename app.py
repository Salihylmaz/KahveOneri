from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import pickle
from datetime import datetime

app = Flask(__name__)

class AIKahveOnericiSistemi:
    def _init_(self):
        self.kahveciler = {}
        self.alerjen_listesi = {
            'sut': 'Süt',
            'kakao': 'Kakao/Çikolata',
            'findik': 'Fındık',
            'antep_fistigi': 'Antep Fıstığı',
            'badem': 'Badem',
            'soya': 'Soya',
            'gluten': 'Gluten'
        }
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.user_preferences_history = []
        self.feedback_file = 'kahve_feedback.csv'
        self.menu_yukle()
        self.ai_model_hazirla()
    
    def menu_yukle(self):
        """CSV dosyalarından kahveci menülerini yükle"""
        csv_dosyalari = {
            'Starbucks': 'starbucks_menu.csv',
            'Mikel Coffee': 'mikel_menu.csv',
            'Gloria Jeans': 'gloria_menu.csv',
            'Coffy': 'coffy_menu.csv'
        }
        
        for kahveci_adi, dosya_adi in csv_dosyalari.items():
            try:
                if os.path.exists(dosya_adi):
                    df = pd.read_csv(dosya_adi)
                    # Özellikler sütununu liste haline getir
                    df['ozellikler'] = df['ozellikler'].apply(lambda x: x.split(',') if pd.notna(x) else [])
                    # Alerjenler sütununu liste haline getir
                    df['alerjenler'] = df['alerjenler'].apply(lambda x: x.split(',') if pd.notna(x) and x.strip() != '' else [])
                    # Kahveci bilgisini ekle
                    df['kahveci'] = kahveci_adi
                    self.kahveciler[kahveci_adi] = df
                    print(f"{kahveci_adi} menüsü yüklendi: {len(df)} ürün")
                else:
                    print(f"Uyarı: {dosya_adi} dosyası bulunamadı")
            except Exception as e:
                print(f"Hata: {kahveci_adi} menüsü yüklenirken hata: {e}")
    
    def ai_model_hazirla(self):
        """AI modeli için veri hazırlama ve eğitim"""
        try:
            # Tüm kahveleri tek bir DataFrame'de birleştir
            all_coffees = []
            for kahveci, df in self.kahveciler.items():
                all_coffees.append(df)
            
            if not all_coffees:
                print("Kahve verisi bulunamadı, AI modeli hazırlanamadı")
                return
            
            combined_df = pd.concat(all_coffees, ignore_index=True)
            
            # Özellik mühendisliği
            feature_df = self.ozellik_muhendisligi(combined_df)
            
            # Simüle edilmiş kullanıcı tercihleri ve puanları oluştur
            training_data = self.simulasyon_verisi_olustur(feature_df)
            
            if len(training_data) > 0:
                # Modeli eğit
                self.model_egit(training_data)
                print("AI modeli başarıyla eğitildi!")
            else:
                print("Eğitim verisi oluşturulamadı")
                
        except Exception as e:
            print(f"AI model hazırlama hatası: {e}")
    
    def ozellik_muhendisligi(self, df):
        """Kahve özelliklerini AI modeli için sayısal verilere dönüştür"""
        feature_df = df.copy()
        
        # Özellik binary encoding
        ozellik_turleri = ['guclu', 'hafif', 'sicak', 'soguk', 'tatli', 'sade']
        for ozellik in ozellik_turleri:
            feature_df[f'has_{ozellik}'] = feature_df['ozellikler'].apply(
                lambda x: 1 if ozellik in x else 0
            )
        
        # Kategori encoding
        if 'kategori' in feature_df.columns:
            le_kategori = LabelEncoder()
            feature_df['kategori_encoded'] = le_kategori.fit_transform(feature_df['kategori'])
            self.label_encoders['kategori'] = le_kategori
        
        # Kahveci encoding
        le_kahveci = LabelEncoder()
        feature_df['kahveci_encoded'] = le_kahveci.fit_transform(feature_df['kahveci'])
        self.label_encoders['kahveci'] = le_kahveci
        
        # Fiyat normalizasyonu (0-1 arası)
        feature_df['fiyat_normalized'] = (feature_df['fiyat'] - feature_df['fiyat'].min()) / (feature_df['fiyat'].max() - feature_df['fiyat'].min())
        
        # Alerjen sayısı
        feature_df['alerjen_sayisi'] = feature_df['alerjenler'].apply(len)
        
        # Model için kullanılacak özellik sütunları
        self.feature_columns = [
            'has_guclu', 'has_hafif', 'has_sicak', 'has_soguk', 'has_tatli', 'has_sade',
            'kategori_encoded', 'kahveci_encoded', 'fiyat_normalized', 'alerjen_sayisi'
        ]
        
        return feature_df
    
    def simulasyon_verisi_olustur(self, df):
        """Kullanıcı tercihleri ve puanlarını simüle et"""
        training_data = []
        
        # Farklı kullanıcı profillerini simüle et
        user_profiles = [
            {'tercihler': ['guclu', 'sicak'], 'olumsuz': ['tatli'], 'weight': 0.8},
            {'tercihler': ['hafif', 'soguk'], 'olumsuz': ['guclu'], 'weight': 0.7},
            {'tercihler': ['tatli', 'sicak'], 'olumsuz': ['sade'], 'weight': 0.9},
            {'tercihler': ['sade', 'guclu'], 'olumsuz': ['tatli'], 'weight': 0.6},
            {'tercihler': ['soguk', 'tatli'], 'olumsuz': ['guclu'], 'weight': 0.8},
            {'tercihler': ['sicak', 'hafif'], 'olumsuz': ['soguk'], 'weight': 0.7},
        ]
        
        for _, kahve in df.iterrows():
            for profile in user_profiles:
                # Kullanıcı tercih vektörü oluştur
                user_vector = self.kullanici_vektoru_olustur(profile['tercihler'])
                
                # Puan hesapla
                score = self.puan_hesapla(kahve, profile)
                
                # Eğitim verisi oluştur
                features = user_vector + [kahve[col] for col in self.feature_columns]
                training_data.append({
                    'features': features,
                    'score': score,
                    'kahve_index': kahve.name
                })
        
        return training_data
    
    def kullanici_vektoru_olustur(self, tercihler):
        """Kullanıcı tercihlerini vektöre dönüştür"""
        ozellik_turleri = ['guclu', 'hafif', 'sicak', 'soguk', 'tatli', 'sade']
        return [1 if ozellik in tercihler else 0 for ozellik in ozellik_turleri]
    
    def puan_hesapla(self, kahve, profile):
        """Kahve ve kullanıcı profili arasında puan hesapla"""
        score = 0.5  # Base score
        
        # Pozitif tercihler
        for tercih in profile['tercihler']:
            if tercih in kahve['ozellikler']:
                score += 0.3
        
        # Negatif tercihler
        for olumsuz in profile.get('olumsuz', []):
            if olumsuz in kahve['ozellikler']:
                score -= 0.2
        
        # Fiyat faktörü (düşük fiyat bonus)
        if kahve['fiyat'] < 25:
            score += 0.1
        elif kahve['fiyat'] > 35:
            score -= 0.1
        
        # Alerjen penalty
        if len(kahve['alerjenler']) > 2:
            score -= 0.1
        
        # Profil ağırlığını uygula
        score *= profile['weight']
        
        # Rastgele gürültü ekle (0.9-1.1 arası)
        score *= (0.9 + random.random() * 0.2)
        
        return max(0, min(1, score))  # 0-1 arası sınırla
    
    def model_egit(self, training_data):
        """Random Forest modelini eğit"""
        if not training_data:
            return
        
        # Veriyi hazırla
        X = np.array([item['features'] for item in training_data])
        y = np.array([item['score'] for item in training_data])
        
        # Random Forest modeli oluştur ve eğit
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.model.fit(X, y)
        
        # Modeli kaydet
        try:
            with open('kahve_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoders': self.label_encoders,
                    'feature_columns': self.feature_columns
                }, f)
        except Exception as e:
            print(f"Model kaydedilirken hata: {e}")
    
    def kahveci_alerjenleri_al(self, kahveci_adi):
        """Belirli bir kahvecinin menüsündeki tüm alerjenleri getir"""
        if kahveci_adi not in self.kahveciler:
            return []
        
        df = self.kahveciler[kahveci_adi]
        tum_alerjenler = set()
        
        for _, kahve in df.iterrows():
            if kahve['alerjenler']:
                tum_alerjenler.update(kahve['alerjenler'])
        
        # Alerjen isimlerini de ekle
        alerjen_listesi = []
        for alerjen_kodu in tum_alerjenler:
            alerjen_listesi.append({
                'kod': alerjen_kodu,
                'isim': self.alerjen_listesi.get(alerjen_kodu, alerjen_kodu)
            })
        
        return sorted(alerjen_listesi, key=lambda x: x['isim'])
    
    def coklu_ai_kahve_onerisi(self, kahveci_adi, tercihler, alerjenler=None, max_oneri=5):
        """AI modeli ile çoklu kahve önerisi - beğeni sırasına göre"""
        if not self.model or kahveci_adi not in self.kahveciler:
            # Fallback to traditional method
            return self.coklu_kahve_onerisi_yap(kahveci_adi, tercihler, alerjenler, max_oneri)
        
        try:
            df = self.kahveciler[kahveci_adi]
            
            # Alerjen filtresi - KATICI FİLTRE
            if alerjenler:
                # Seçilen alerjenlerin hiçbirini içermeyen kahveleri getir
                filtered_df = df[~df['alerjenler'].apply(
                    lambda x: any(alerjen in x for alerjen in alerjenler)
                )]
            else:
                filtered_df = df
            
            if len(filtered_df) == 0:
                return {
                    'hata': 'Seçtiğiniz alerjilere uygun kahve bulunamadı!',
                    'alerjenler': [self.alerjen_listesi.get(a, a) for a in alerjenler] if alerjenler else []
                }
            
            # Kullanıcı tercih vektörü
            user_vector = self.kullanici_vektoru_olustur(tercihler)
            
            # Her kahve için AI puanı hesapla
            coffee_scores = []
            for idx, kahve in filtered_df.iterrows():
                # Kahve özelliklerini hazırla
                coffee_features = self.kahve_ozelliklerini_hazirla(kahve)
                
                # Tam feature vektörü
                full_features = user_vector + coffee_features
                
                # AI puanı tahmin et
                try:
                    score = self.model.predict([full_features])[0]
                    
                    # Tercih uyumluluk bonusu
                    tercih_bonus = sum(1 for tercih in tercihler if tercih in kahve['ozellikler']) * 0.1
                    score += tercih_bonus
                    
                    coffee_scores.append((idx, score, kahve.to_dict()))
                except Exception as e:
                    print(f"Tahmin hatası: {e}")
                    # Fallback score
                    score = random.random() * 0.5 + 0.25
                    coffee_scores.append((idx, score, kahve.to_dict()))
            
            # Puanına göre sırala (yüksekten düşüğe)
            coffee_scores.sort(key=lambda x: x[1], reverse=True)
            
            # En iyi önerileri al
            top_recommendations = []
            for i, (idx, confidence, kahve_dict) in enumerate(coffee_scores[:max_oneri]):
                # Güven skoru ve sıra bilgisi ekle
                kahve_dict['ai_confidence'] = round(confidence * 100, 1)
                kahve_dict['rank'] = i + 1
                kahve_dict['recommendation_reason'] = self.oneri_gerekce_olustur(
                    kahve_dict, tercihler, confidence, i + 1
                )
                
                # Alerjen isimlerini ekle
                if kahve_dict['alerjenler']:
                    kahve_dict['alerjen_isimleri'] = [
                        self.alerjen_listesi.get(alerjen, alerjen) 
                        for alerjen in kahve_dict['alerjenler']
                    ]
                else:
                    kahve_dict['alerjen_isimleri'] = []
                
                top_recommendations.append(kahve_dict)
            
            # Kullanıcı tercihlerini kaydet
            if top_recommendations:
                self.kullanici_tercihi_kaydet(tercihler, alerjenler, top_recommendations[0])
            
            return {
                'oneriler': top_recommendations,
                'toplam_oneri': len(top_recommendations),
                'filtrelenen_urun_sayisi': len(filtered_df),
                'toplam_urun_sayisi': len(df)
            }
            
        except Exception as e:
            print(f"AI çoklu öneri hatası: {e}")
            return self.coklu_kahve_onerisi_yap(kahveci_adi, tercihler, alerjenler, max_oneri)
    
    def kahve_ozelliklerini_hazirla(self, kahve):
        """Kahve özelliklerini model için hazırla"""
        features = []
        
        # Özellik binary encoding
        ozellik_turleri = ['guclu', 'hafif', 'sicak', 'soguk', 'tatli', 'sade']
        for ozellik in ozellik_turleri:
            features.append(1 if ozellik in kahve['ozellikler'] else 0)
        
        # Kategori encoding
        try:
            if 'kategori' in self.label_encoders:
                kategori_encoded = self.label_encoders['kategori'].transform([kahve['kategori']])[0]
            else:
                kategori_encoded = 0
        except:
            kategori_encoded = 0
        features.append(kategori_encoded)
        
        # Kahveci encoding
        try:
            if 'kahveci' in self.label_encoders:
                kahveci_encoded = self.label_encoders['kahveci'].transform([kahve['kahveci']])[0]
            else:
                kahveci_encoded = 0
        except:
            kahveci_encoded = 0
        features.append(kahveci_encoded)
        
        # Fiyat normalizasyonu (rough estimation)
        fiyat_normalized = min(1, max(0, (kahve['fiyat'] - 15) / 30))
        features.append(fiyat_normalized)
        
        # Alerjen sayısı
        features.append(len(kahve['alerjenler']))
        
        return features
    
    def oneri_gerekce_olustur(self, kahve, tercihler, confidence, rank):
        """AI önerisi için gerekçe oluştur"""
        reasons = []
        
        # Sıra bilgisi
        if rank == 1:
            reasons.append("En çok beğeneceğiniz kahve")
        elif rank <= 3:
            reasons.append(f"{rank}. en iyi seçenek")
        
        # Tercih eşleşmeleri
        matched_preferences = [t for t in tercihler if t in kahve['ozellikler']]
        if matched_preferences:
            if len(matched_preferences) == 1:
                reasons.append(f"'{matched_preferences[0]}' tercihiniz ile uyumlu")
            else:
                reasons.append(f"Tercihleriniz ({', '.join(matched_preferences)}) ile uyumlu")
        
        # Güven skoru
        if confidence > 0.8:
            reasons.append("Yüksek güvenle öneriliyor")
        elif confidence > 0.6:
            reasons.append("Orta güvenle öneriliyor")
        
        # Fiyat faktörü
        if kahve['fiyat'] < 25:
            reasons.append("Uygun fiyatlı")
        elif kahve['fiyat'] > 35:
            reasons.append("Premium seçenek")
        
        # Alerjen durumu
        if not kahve['alerjenler']:
            reasons.append("Alerjen içermiyor")
        
        return " • ".join(reasons) if reasons else "AI algoritması tarafından önerildi"
    
    def coklu_kahve_onerisi_yap(self, kahveci_adi, tercihler, alerjenler=None, max_oneri=5):
        """Geleneksel çoklu kahve önerisi (fallback)"""
        if kahveci_adi not in self.kahveciler:
            return {'hata': 'Kahveci bulunamadı!'}
        
        df = self.kahveciler[kahveci_adi]
        
        # Alerjen filtresi - KATICI FİLTRE
        if alerjenler:
            alerjen_filtreli_df = df[~df['alerjenler'].apply(
                lambda x: any(alerjen in x for alerjen in alerjenler)
            )]
        else:
            alerjen_filtreli_df = df
        
        if len(alerjen_filtreli_df) == 0:
            return {
                'hata': 'Seçtiğiniz alerjilere uygun kahve bulunamadı!',
                'alerjenler': [self.alerjen_listesi.get(a, a) for a in alerjenler] if alerjenler else []
            }
        
        # Tercih puanlaması
        scored_coffees = []
        for index, kahve in alerjen_filtreli_df.iterrows():
            kahve_dict = kahve.to_dict()
            
            # Puan hesapla
            score = 0
            matched_preferences = []
            
            # Tercih eşleşmesi
            for tercih in tercihler:
                if tercih in kahve['ozellikler']:
                    score += 1
                    matched_preferences.append(tercih)
            
            # Fiyat bonusu
            if kahve['fiyat'] < 25:
                score += 0.5
            
            # Alerjen penalty (az alerjen = bonus)
            if len(kahve['alerjenler']) == 0:
                score += 0.3
            elif len(kahve['alerjenler']) <= 1:
                score += 0.1
            
            # Rastgele faktör (çeşitlilik için)
            score += random.random() * 0.3
            
            # Ekstra bilgiler
            kahve_dict['score'] = score
            kahve_dict['matched_preferences'] = matched_preferences
            kahve_dict['method'] = 'traditional'
            
            # Alerjen isimlerini ekle
            if kahve_dict['alerjenler']:
                kahve_dict['alerjen_isimleri'] = [
                    self.alerjen_listesi.get(alerjen, alerjen) 
                    for alerjen in kahve_dict['alerjenler']
                ]
            else:
                kahve_dict['alerjen_isimleri'] = []
            
            scored_coffees.append(kahve_dict)
        
        # Puana göre sırala
        scored_coffees.sort(key=lambda x: x['score'], reverse=True)
        
        # En iyi önerileri al
        top_recommendations = []
        for i, kahve in enumerate(scored_coffees[:max_oneri]):
            kahve['rank'] = i + 1
            kahve['confidence'] = min(100, kahve['score'] * 30)  # Basit güven skoru
            
            # Gerekçe oluştur
            reasons = []
            if kahve['rank'] == 1:
                reasons.append("En iyi eşleşme")
            elif kahve['rank'] <= 3:
                reasons.append(f"{kahve['rank']}. en iyi seçenek")
            
            if kahve['matched_preferences']:
                reasons.append(f"Tercihleriniz ile uyumlu: {', '.join(kahve['matched_preferences'])}")
            
            if kahve['fiyat'] < 25:
                reasons.append("Ekonomik seçenek")
            
            if not kahve['alerjenler']:
                reasons.append("Alerjen içermiyor")
            
            kahve['recommendation_reason'] = " • ".join(reasons) if reasons else "Size özel seçim"
            
            top_recommendations.append(kahve)
        
        return {
            'oneriler': top_recommendations,
            'toplam_oneri': len(top_recommendations),
            'filtrelenen_urun_sayisi': len(alerjen_filtreli_df),
            'toplam_urun_sayisi': len(df)
        }
    
    def kullanici_tercihi_kaydet(self, tercihler, alerjenler, secilen_kahve):
        """Kullanıcı tercihlerini gelecek öneriler için kaydet"""
        self.user_preferences_history.append({
            'timestamp': datetime.now().isoformat(),
            'tercihler': tercihler,
            'alerjenler': alerjenler or [],
            'secilen_kahve': secilen_kahve['kahve_adi'],
            'kahveci': secilen_kahve['kahveci'],
            'fiyat': secilen_kahve['fiyat'],
            'ai_confidence': secilen_kahve.get('ai_confidence', 0)
        })
        
        # Sadece son 100 kaydı tut
        if len(self.user_preferences_history) > 100:
            self.user_preferences_history = self.user_preferences_history[-100:]
    
    def kahveci_listesi_al(self):
        return list(self.kahveciler.keys())
    
    def alerjen_listesi_al(self):
        return self.alerjen_listesi
    
    def kahveci_menusu_al(self, kahveci_adi, alerjenler=None):
        if kahveci_adi not in self.kahveciler:
            return []
            
        df = self.kahveciler[kahveci_adi]
        
        if alerjenler:
            df = df[~df['alerjenler'].apply(lambda x: any(alerjen in x for alerjen in alerjenler))]
        
        menu_listesi = df.to_dict('records')
        
        for item in menu_listesi:
            if item['alerjenler']:
                item['alerjen_isimleri'] = [self.alerjen_listesi.get(alerjen, alerjen) for alerjen in item['alerjenler']]
            else:
                item['alerjen_isimleri'] = []
                
        return menu_listesi
    
    def ai_istatistikler(self):
        """AI modeli istatistikleri"""
        stats = {
            'model_active': self.model is not None,
            'total_preferences_recorded': len(self.user_preferences_history),
            'feature_count': len(self.feature_columns),
            'label_encoders': list(self.label_encoders.keys()),
            'recent_recommendations': self.user_preferences_history[-5:] if self.user_preferences_history else []
        }
        
        if self.model:
            try:
                # Feature importance
                feature_names = ['user_' + f for f in ['guclu', 'hafif', 'sicak', 'soguk', 'tatli', 'sade']] + self.feature_columns
                importances = self.model.feature_importances_
                stats['feature_importance'] = dict(zip(feature_names, importances.tolist()))
            except:
                stats['feature_importance'] = {}
        
        return stats

    def gunun_kahvesi_sec(self):
        """Günün kahvesini seç"""
        try:
            # Tüm kahveleri birleştir
            tum_kahveler = []
            for kahveci, df in self.kahveciler.items():
                for _, kahve in df.iterrows():
                    kahve_dict = kahve.to_dict()
                    kahve_dict['kahveci'] = kahveci
                    tum_kahveler.append(kahve_dict)
            
            if not tum_kahveler:
                return None
            
            # Günün tarihini seed olarak kullan
            bugun = datetime.now().date()
            random.seed(bugun.toordinal())
            
            # Rastgele bir kahve seç
            secilen_kahve = random.choice(tum_kahveler)
            
            # Alerjen isimlerini ekle
            if secilen_kahve['alerjenler']:
                secilen_kahve['alerjen_isimleri'] = [
                    self.alerjen_listesi.get(alerjen, alerjen) 
                    for alerjen in secilen_kahve['alerjenler']
                ]
            else:
                secilen_kahve['alerjen_isimleri'] = []
            
            return secilen_kahve
            
        except Exception as e:
            print(f"Günün kahvesi seçilirken hata: {e}")
            return None

    def feedback_kaydet(self, kahve_adi, kahveci, tercihler, alerjenler, begeni_puani, yorum):
        """Kahve önerisi geri bildirimini kaydet"""
        try:
            # CSV dosyası yoksa oluştur
            if not os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'w', encoding='utf-8') as f:
                    f.write('timestamp,kahve_adi,kahveci,tercihler,alerjenler,beğeni_puanı,yorum\n')
            
            # Yeni feedback'i ekle
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                tercihler_str = ','.join(tercihler) if tercihler else ''
                alerjenler_str = ','.join(alerjenler) if alerjenler else ''
                f.write(f'{timestamp},{kahve_adi},{kahveci},"{tercihler_str}","{alerjenler_str}",{begeni_puani},"{yorum}"\n')
            
            return True
            
        except Exception as e:
            print(f"Feedback kaydedilirken hata: {e}")
            return False

    def feedback_istatistikleri(self):
        """Feedback istatistiklerini getir"""
        try:
            if not os.path.exists(self.feedback_file):
                return {
                    'toplam_feedback': 0,
                    'ortalama_puan': 0,
                    'en_cok_begenilen': None,
                    'son_feedbackler': []
                }
            
            df = pd.read_csv(self.feedback_file)
            
            if len(df) == 0:
                return {
                    'toplam_feedback': 0,
                    'ortalama_puan': 0,
                    'en_cok_begenilen': None,
                    'son_feedbackler': []
                }
            
            # En çok beğenilen kahve
            en_cok_begenilen = df.loc[df['beğeni_puanı'].idxmax()]
            
            # Son 5 feedback
            son_feedbackler = df.tail(5).to_dict('records')
            
            return {
                'toplam_feedback': len(df),
                'ortalama_puan': round(df['beğeni_puanı'].mean(), 2),
                'en_cok_begenilen': {
                    'kahve_adi': en_cok_begenilen['kahve_adi'],
                    'kahveci': en_cok_begenilen['kahveci'],
                    'puan': en_cok_begenilen['beğeni_puanı']
                },
                'son_feedbackler': son_feedbackler
            }
            
        except Exception as e:
            print(f"Feedback istatistikleri alınırken hata: {e}")
            return {
                'toplam_feedback': 0,
                'ortalama_puan': 0,
                'en_cok_begenilen': None,
                'son_feedbackler': []
            }

# Global AI kahve önerici sistemi
ai_kahve_sistemi = AIKahveOnericiSistemi()

@app.route('/')
def ana_sayfa():
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
        '''
    return render_template('index.html')

@app.route('/kahveciler')
def kahveciler():
    return jsonify(ai_kahve_sistemi.kahveci_listesi_al())

@app.route('/alerjenler')
def alerjenler():
    return jsonify(ai_kahve_sistemi.alerjen_listesi_al())

@app.route('/kahveci-alerjenleri/<kahveci_adi>')
def kahveci_alerjenleri(kahveci_adi):
    """Belirli kahvecinin menüsündeki alerjenleri getir"""
    alerjenler = ai_kahve_sistemi.kahveci_alerjenleri_al(kahveci_adi)
    return jsonify(alerjenler)

@app.route('/menu/<kahveci_adi>')
def kahveci_menusu(kahveci_adi):
    alerjenler = request.args.get('alerjenler', '').split(',') if request.args.get('alerjenler') else None
    if alerjenler and alerjenler[0] == '':
        alerjenler = None
    menu = ai_kahve_sistemi.kahveci_menusu_al(kahveci_adi, alerjenler)
    return jsonify(menu)

@app.route('/coklu-ai-oneri', methods=['POST'])
def coklu_ai_kahve_onerisi():
    """AI destekli çoklu kahve önerisi endpoint"""
    try:
        veri = request.get_json()
        if not veri:
            return jsonify({'hata': 'JSON verisi bulunamadı!'}), 400

        kahveci_adi = veri.get('kahveci')
        tercihler = veri.get('tercihler', [])
        alerjenler = veri.get('alerjenler', [])
        max_oneri = veri.get('max_oneri', 5)

        if not kahveci_adi:
            return jsonify({'hata': 'Kahveci seçilmedi!'}), 400

        if not tercihler:
            return jsonify({'hata': 'En az bir tercih seçilmelidir!'}), 400

        # Kahveci kontrolü
        if kahveci_adi not in ai_kahve_sistemi.kahveciler:
            return jsonify({
                'hata': 'Geçersiz kahveci!',
                'mevcut_kahveciler': list(ai_kahve_sistemi.kahveciler.keys())
            }), 400

        # AI ile çoklu öneri al
        oneriler = ai_kahve_sistemi.coklu_ai_kahve_onerisi(
            kahveci_adi=kahveci_adi,
            tercihler=tercihler,
            alerjenler=alerjenler,
            max_oneri=max_oneri
        )

        if 'hata' in oneriler:
            return jsonify(oneriler), 400

        return jsonify(oneriler)

    except Exception as e:
        import traceback
        hata_detay = traceback.format_exc()
        print(f"Hata detayı: {hata_detay}")
        return jsonify({
            'hata': f'Öneri oluşturulurken hata: {str(e)}',
            'hata_detay': hata_detay
        }), 500

@app.route('/gunun-kahvesi')
def gunun_kahvesi():
    """Günün kahvesini getir"""
    kahve = ai_kahve_sistemi.gunun_kahvesi_sec()
    if kahve:
        return jsonify(kahve)
    return jsonify({'hata': 'Günün kahvesi seçilemedi!'}), 500

@app.route('/feedback', methods=['POST'])
def feedback_kaydet():
    """Kahve önerisi geri bildirimini kaydet"""
    try:
        veri = request.get_json()
        kahve_adi = veri.get('kahve_adi')
        kahveci = veri.get('kahveci')
        tercihler = veri.get('tercihler', [])
        alerjenler = veri.get('alerjenler', [])
        begeni_puani = veri.get('begeni_puani')
        yorum = veri.get('yorum', '')

        if not all([kahve_adi, kahveci, begeni_puani]):
            return jsonify({'hata': 'Eksik bilgi!'}), 400

        if not isinstance(begeni_puani, (int, float)) or begeni_puani < 1 or begeni_puani > 5:
            return jsonify({'hata': 'Geçersiz beğeni puanı! (1-5 arası olmalı)'}), 400

        basarili = ai_kahve_sistemi.feedback_kaydet(
            kahve_adi=kahve_adi,
            kahveci=kahveci,
            tercihler=tercihler,
            alerjenler=alerjenler,
            begeni_puani=begeni_puani,
            yorum=yorum
        )

        if basarili:
            return jsonify({'mesaj': 'Geri bildiriminiz kaydedildi!'})
        return jsonify({'hata': 'Geri bildirim kaydedilemedi!'}), 500

    except Exception as e:
        return jsonify({'hata': f'Geri bildirim kaydedilirken hata: {str(e)}'}), 500

@app.route('/feedback-istatistikleri')
def feedback_istatistikleri():
    """Feedback istatistiklerini getir"""
    istatistikler = ai_kahve_sistemi.feedback_istatistikleri()
    return jsonify(istatistikler)

if __name__ == '__main__':
    app.run(debug=True)
