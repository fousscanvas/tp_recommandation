"""
Catalogue de produits pour le micro e-commerce.
Chaque produit a: id, name, category, price, image_url, description
"""

PRODUCTS = [
    {
        'id': 0,
        'name': 'Laptop Pro',
        'category': 'electronics',
        'price': 999.99,
        'image': 'https://source.unsplash.com/400x300/?laptop,computer',
        'description': 'Portable haute performance'
    },
    {
        'id': 1,
        'name': 'Wireless Headphones',
        'category': 'electronics',
        'price': 199.99,
        'image': 'https://source.unsplash.com/400x300/?headphones,audio',
        'description': 'Son premium sans fil'
    },
    {
        'id': 2,
        'name': 'USB-C Cable',
        'category': 'electronics',
        'price': 29.99,
        'image': 'https://source.unsplash.com/400x300/?cable,connector',
        'description': 'Câble haute vitesse'
    },
    {
        'id': 3,
        'name': 'Sneakers Running',
        'category': 'clothing',
        'price': 89.99,
        'image': 'https://source.unsplash.com/400x300/?sneakers,running,shoes',
        'description': 'Chaussures de course confortables'
    },
    {
        'id': 4,
        'name': 'T-Shirt Cotton',
        'category': 'clothing',
        'price': 24.99,
        'image': 'https://source.unsplash.com/400x300/?tshirt,cotton,clothing',
        'description': 'T-shirt 100% coton'
    },
    {
        'id': 5,
        'name': 'Jeans Classic',
        'category': 'clothing',
        'price': 69.99,
        'image': 'https://source.unsplash.com/400x300/?jeans,denim',
        'description': 'Jean durable et élégant'
    },
    {
        'id': 6,
        'name': 'Coffee Maker',
        'category': 'home',
        'price': 79.99,
        'image': 'https://source.unsplash.com/400x300/?coffee,maker,kitchen',
        'description': 'Machine à café programmable'
    },
    {
        'id': 7,
        'name': 'Desk Lamp',
        'category': 'home',
        'price': 49.99,
        'image': 'https://source.unsplash.com/400x300/?lamp,desk,lighting',
        'description': 'Lampe LED ajustable'
    },
    {
        'id': 8,
        'name': 'Plant Pot',
        'category': 'home',
        'price': 19.99,
        'image': 'https://source.unsplash.com/400x300/?plant,pot,decor',
        'description': 'Pot élégant pour plantes'
    },
    {
        'id': 9,
        'name': 'Book Python',
        'category': 'books',
        'price': 39.99,
        'image': 'https://source.unsplash.com/400x300/?python,programming,book',
        'description': 'Guide complet Python'
    },
    # === ELECTRONICS (10-17) ===
    {
        'id': 10,
        'name': 'Smartphone Pro',
        'category': 'electronics',
        'price': 799.99,
        'image': 'https://source.unsplash.com/400x300/?smartphone,mobile,phone',
        'description': 'Téléphone haut de gamme'
    },
    {
        'id': 11,
        'name': 'Tablet Plus',
        'category': 'electronics',
        'price': 599.99,
        'image': 'https://source.unsplash.com/400x300/?tablet,ipad,device',
        'description': 'Tablette tactile haute performance'
    },
    {
        'id': 12,
        'name': '4K Monitor',
        'category': 'electronics',
        'price': 449.99,
        'image': 'https://source.unsplash.com/400x300/?monitor,screen,4k',
        'description': 'Écran 4K 27 pouces'
    },
    {
        'id': 13,
        'name': 'Mechanical Keyboard',
        'category': 'electronics',
        'price': 149.99,
        'image': 'https://source.unsplash.com/400x300/?keyboard,mechanical,rgb',
        'description': 'Clavier mécanique RGB'
    },
    {
        'id': 14,
        'name': 'Gaming Mouse',
        'category': 'electronics',
        'price': 79.99,
        'image': 'https://source.unsplash.com/400x300/?mouse,gaming,computer',
        'description': 'Souris gaming précise'
    },
    {
        'id': 15,
        'name': 'Webcam HD',
        'category': 'electronics',
        'price': 99.99,
        'image': 'https://source.unsplash.com/400x300/?webcam,camera,video',
        'description': 'Webcam 1080p avec micro'
    },
    {
        'id': 16,
        'name': 'Power Bank 20K',
        'category': 'electronics',
        'price': 44.99,
        'image': 'https://source.unsplash.com/400x300/?powerbank,battery,charger',
        'description': 'Batterie externe 20000mAh'
    },
    {
        'id': 17,
        'name': 'USB Hub 7 Ports',
        'category': 'electronics',
        'price': 34.99,
        'image': 'https://source.unsplash.com/400x300/?usb,hub,connector',
        'description': 'Hub USB 3.0 haute vitesse'
    },
    # === CLOTHING (18-25) ===
    {
        'id': 18,
        'name': 'Winter Jacket',
        'category': 'clothing',
        'price': 129.99,
        'image': 'https://source.unsplash.com/400x300/?winter,jacket,coat',
        'description': 'Veste d\'hiver chaude et élégante'
    },
    {
        'id': 19,
        'name': 'Casual Dress',
        'category': 'clothing',
        'price': 59.99,
        'image': 'https://source.unsplash.com/400x300/?dress,casual,fashion',
        'description': 'Robe décontractée chic'
    },
    {
        'id': 20,
        'name': 'Leather Belt',
        'category': 'clothing',
        'price': 34.99,
        'image': 'https://source.unsplash.com/400x300/?belt,leather,accessories',
        'description': 'Ceinture cuir véritable'
    },
    {
        'id': 21,
        'name': 'Socks Pack',
        'category': 'clothing',
        'price': 14.99,
        'image': 'https://source.unsplash.com/400x300/?socks,fabric,apparel',
        'description': 'Pack de 5 paires de chaussettes'
    },
    {
        'id': 22,
        'name': 'Baseball Cap',
        'category': 'clothing',
        'price': 24.99,
        'image': 'https://source.unsplash.com/400x300/?cap,baseball,hat',
        'description': 'Casquette baseball ajustable'
    },
    {
        'id': 23,
        'name': 'Sports Shorts',
        'category': 'clothing',
        'price': 39.99,
        'image': 'https://source.unsplash.com/400x300/?shorts,sports,gym',
        'description': 'Short de sport respirant'
    },
    {
        'id': 24,
        'name': 'Polo Shirt',
        'category': 'clothing',
        'price': 44.99,
        'image': 'https://source.unsplash.com/400x300/?polo,shirt,casual',
        'description': 'Polo classique confortable'
    },
    {
        'id': 25,
        'name': 'Scarf Wool',
        'category': 'clothing',
        'price': 29.99,
        'image': 'https://source.unsplash.com/400x300/?scarf,wool,winter',
        'description': 'Écharpe laine premium'
    },
    # === HOME (26-32) ===
    {
        'id': 26,
        'name': 'Bath Towel',
        'category': 'home',
        'price': 24.99,
        'image': 'https://source.unsplash.com/400x300/?towel,bath,textile',
        'description': 'Serviette de bain douce et absorbante'
    },
    {
        'id': 27,
        'name': 'Pillow Memory Foam',
        'category': 'home',
        'price': 59.99,
        'image': 'https://source.unsplash.com/400x300/?pillow,foam,bed',
        'description': 'Oreiller mousse à mémoire'
    },
    {
        'id': 28,
        'name': 'Bed Sheet Set',
        'category': 'home',
        'price': 79.99,
        'image': 'https://source.unsplash.com/400x300/?bedsheet,linen,cotton',
        'description': 'Ensemble draps coton 100%'
    },
    {
        'id': 29,
        'name': 'Kitchen Knife Set',
        'category': 'home',
        'price': 89.99,
        'image': 'https://source.unsplash.com/400x300/?knife,kitchen,cooking',
        'description': 'Ensemble 5 couteaux de cuisine'
    },
    {
        'id': 30,
        'name': 'Storage Box',
        'category': 'home',
        'price': 34.99,
        'image': 'https://source.unsplash.com/400x300/?storage,box,organize',
        'description': 'Boîte de rangement modulable'
    },
    {
        'id': 31,
        'name': 'Candle Scented',
        'category': 'home',
        'price': 19.99,
        'image': 'https://source.unsplash.com/400x300/?candle,scented,aromatherapy',
        'description': 'Bougie parfumée lavande'
    },
    {
        'id': 32,
        'name': 'Bed Frame Wood',
        'category': 'home',
        'price': 299.99,
        'image': 'https://source.unsplash.com/400x300/?bed,frame,wood,furniture',
        'description': 'Cadre de lit en bois massif'
    },
    # === BOOKS (33-35) ===
    {
        'id': 33,
        'name': 'Book JavaScript',
        'category': 'books',
        'price': 34.99,
        'image': 'https://source.unsplash.com/400x300/?javascript,programming,code',
        'description': 'Maîtriser le JavaScript moderne'
    },
    {
        'id': 34,
        'name': 'Book Machine Learning',
        'category': 'books',
        'price': 49.99,
        'image': 'https://source.unsplash.com/400x300/?machine,learning,ai,data',
        'description': 'Introduction au Machine Learning'
    },
    {
        'id': 35,
        'name': 'Book Design Patterns',
        'category': 'books',
        'price': 44.99,
        'image': 'https://source.unsplash.com/400x300/?design,patterns,code',
        'description': 'Patterns de conception pour tous'
    }
]

# Mapping catégories → indices pour encoding
CATEGORY_MAP = {
    'electronics': 0,
    'clothing': 1,
    'home': 2,
    'books': 3
}

CATEGORY_INV = {v: k for k, v in CATEGORY_MAP.items()}
N_ITEMS = len(PRODUCTS)
N_CATEGORIES = len(CATEGORY_MAP)


def get_product_by_id(pid):
    """Récupère un produit par ID"""
    if 0 <= pid < N_ITEMS:
        return PRODUCTS[pid]
    return None


def get_category_encoded(category_name):
    """Encode catégorie en nombre"""
    return CATEGORY_MAP.get(category_name, 0)


def get_price_normalized(price):
    """Normalise le prix entre [0, 1]. Max price = 1000$"""
    return min(price / 1000.0, 1.0)


if __name__ == '__main__':
    # Test rapide
    print(f"Total produits: {N_ITEMS}")
    print(f"Total catégories: {N_CATEGORIES}")
    print(f"Produit 0: {PRODUCTS[0]}")
    print(f"Catégorie encoded 'electronics': {get_category_encoded('electronics')}")
    print(f"Prix normalized 999: {get_price_normalized(999)}")
