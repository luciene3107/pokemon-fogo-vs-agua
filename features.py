import warnings
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2hsv, rgb2gray, rgb2lab
from sklearn.cluster import KMeans

SIZE = 128   # todas as imagens viram 128x128

def _abrir(img):
    # Aceita caminho de arquivo ou imagem PIL. Converte para RGBA (mantém a transparência)
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    return img.convert('RGBA').resize((SIZE, SIZE))

def _mascara(rgb, alpha):
    # Quais pixels são o Pokémon (e não o fundo)?
    if alpha.min() > 0.99:
        # Imagem opaca (JPG, sprite com fundo branco): ignora os pixels quase brancos
        mask = ~((rgb > 0.94).all(axis=-1))
    else:
        # Imagem com transparência: usa o canal alpha
        mask = alpha > 0.5
    if mask.sum() < 50:
        mask = np.ones(alpha.shape, dtype=bool)   # segurança: se a máscara ficar vazia, usa tudo
    return mask

def feats_cor(img):
    a = np.asarray(img, dtype=np.float32) / 255.0
    rgb, alpha = a[..., :3], a[..., 3]
    mask = _mascara(rgb, alpha)
    hsv = rgb2hsv(rgb)
    h, s, v = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]
    n = float(mask.sum())
    # Matiz só dos pixels com cor de verdade (ignora branco, cinza e preto, que não têm matiz útil)
    colorido = (s > 0.25) & (v > 0.2)
    hist_h = np.histogram(h[colorido], bins=18, range=(0, 1))[0] / n   # 18 faixas de matiz
    hist_s = np.histogram(s, bins=4, range=(0, 1))[0] / n              # 4 faixas de saturação
    hist_v = np.histogram(v, bins=4, range=(0, 1))[0] / n              # 4 faixas de brilho
    medias = np.concatenate([rgb[mask].mean(axis=0), [s.mean(), v.mean()]])
    return np.concatenate([hist_h, hist_s, hist_v, medias])           # 18 + 4 + 4 + 5 = 31 números

def feats_cor2(img):
    # Cores extras: LAB (percepção humana de cor), matiz médio circular e 3 cores dominantes
    a = np.asarray(img, dtype=np.float32) / 255.0
    rgb, alpha = a[..., :3], a[..., 3]
    mask = _mascara(rgb, alpha)
    px = rgb[mask]                                              # só os pixels do Pokémon, formato (n, 3)
    lab = rgb2lab(px.reshape(-1, 1, 3)).reshape(-1, 3)
    lab_stats = np.concatenate([lab.mean(axis=0), lab.std(axis=0)])            # 6 números
    hsv = rgb2hsv(px.reshape(-1, 1, 3)).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    peso = s * (v > 0.2)                                        # matiz pesado pela saturação
    ang = 2 * np.pi * h                                         # matiz é circular: vermelho fica nas duas pontas
    sen = (peso * np.sin(ang)).sum() / (peso.sum() + 1e-6)
    cos = (peso * np.cos(ang)).sum() / (peso.sum() + 1e-6)
    circ = [sen, cos, np.hypot(sen, cos)]                       # 3 números (direção e força do matiz médio)
    # 3 cores dominantes por k-means (em no máximo 2000 pixels, para ser rápido e reproduzível)
    rng = np.random.RandomState(0)
    amostra = px if len(px) <= 2000 else px[rng.choice(len(px), 2000, replace=False)]
    if len(amostra) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            km = KMeans(n_clusters=3, n_init=3, random_state=0).fit(amostra)
        cont = np.bincount(km.labels_, minlength=3).astype(float)
        ordem = np.argsort(-cont)                               # da cor mais frequente para a menos frequente
        dom = np.concatenate([[cont[i] / cont.sum()] + list(km.cluster_centers_[i]) for i in ordem])   # 3 x 4 = 12
    else:
        dom = np.zeros(12)
    return np.concatenate([lab_stats, circ, dom])               # 6 + 3 + 12 = 21 números

def feats_forma(img):
    a = np.asarray(img, dtype=np.float32) / 255.0
    rgb, alpha = a[..., :3], a[..., 3:4]
    sobre_branco = rgb * alpha + (1 - alpha)       # coloca o Pokémon sobre fundo branco
    gray = rgb2gray(sobre_branco)
    # HOG: direção das bordas em células de 32x32 pixels, normalizada em blocos de 2x2 células.
    # Células grandes geram poucas features (324), para a cor não ficar diluída
    return hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2))

def extrair(img, conjunto='cor+hog'):
    # conjunto: combinação de 'cor', 'cor2' e 'hog' separados por '+', ex.: 'cor+cor2+hog'
    nomes = conjunto.split('+')
    im = _abrir(img)
    partes = []
    if 'cor' in nomes:
        partes.append(feats_cor(im))
    if 'cor2' in nomes:
        partes.append(feats_cor2(im))
    if 'hog' in nomes:
        partes.append(feats_forma(im))
    return np.concatenate(partes)
