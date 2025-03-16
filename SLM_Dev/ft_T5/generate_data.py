import json
import random

# Définition des catégories et des templates d'exemples
categories = ["pua", "promesse", "cession", "protocole", "bails", "garantie", "acte réitératif", "pacte", "transaction"]

templates = {
    "pua": {
       "input": [
           "J'ai besoin d'un PUA pour {target} mais c'est pas clair.",
           "Demande de PUA pour l'acquisition de {target}, c'est un peu brouillon."
       ],
       "output": [
           "Obtenir les modalités d'un PUA concernant {target} avec précision sur les conditions.",
           "Clarifier la demande de PUA pour {target} en détaillant les clauses et engagements."
       ]
    },
    "promesse": {
       "input": [
           "Promesse de vente pour {target} mais ma demande est floue.",
           "Besoin d'une promesse de vente pour {target} sans trop savoir comment formuler."
       ],
       "output": [
           "Formaliser une promesse de vente pour {target} en précisant les engagements réciproques.",
           "Définir clairement une promesse de vente pour {target} avec les conditions contractuelles précises."
       ]
    },
    "cession": {
       "input": [
           "Cession partielle de {target} – demande pas très nette.",
           "Je souhaite céder une part de {target} mais c'est un peu confus."
       ],
       "output": [
           "Clarifier les modalités de cession partielle de {target} en précisant le pourcentage et les conditions.",
           "Décrire de manière précise la cession de parts concernant {target} avec tous les détails requis."
       ]
    },
    "protocole": {
       "input": [
           "Il y a un protocole d'accord pour {target} en préparation, écrit à la va-vite.",
           "Demande de protocole pour {target} avec une formulation qui laisse à désirer."
       ],
       "output": [
           "Rédiger un protocole d'accord pour {target} en précisant les engagements et modalités de collaboration.",
           "Formaliser un protocole d'accord pour {target} avec une description détaillée des clauses."
       ]
    },
    "bails": {
       "input": [
           "J'ai besoin d'un bail pour {target}, mais ma demande est incomplète.",
           "Demande de bail commercial pour {target} formulée de façon confuse."
       ],
       "output": [
           "Obtenir les conditions détaillées d'un bail commercial pour {target}, incluant durée, loyer et obligations.",
           "Clarifier les modalités d'un bail commercial relatif à {target} avec une description précise des conditions contractuelles."
       ]
    },
    "garantie": {
       "input": [
           "Besoin d'une garantie bancaire pour {target}, c'est très brouillon.",
           "Demande de garantie pour {target} formulée de manière imprécise."
       ],
       "output": [
           "Définir les modalités d'une garantie bancaire pour {target} en précisant conditions et critères d'éligibilité.",
           "Clarifier la demande de garantie bancaire pour {target} en détaillant les conditions et les exigences."
       ]
    },
    "acte réitératif": {
       "input": [
           "Je veux un acte réitératif pour {target} mais c'est un peu confus.",
           "Demande d'acte réitératif pour {target} sans formulation précise."
       ],
       "output": [
           "Élaborer un acte réitératif pour {target} en précisant les modalités de renouvellement automatique.",
           "Formaliser un acte réitératif pour {target} avec une description claire des conditions de renouvellement."
       ]
    },
    "pacte": {
       "input": [
           "Besoin d'un pacte d'associés pour régir l'entrée d'un investisseur dans {target} mais c'est mal rédigé.",
           "Je veux un pacte pour l'entrée d'un investisseur dans {target} et c'est pas très clair."
       ],
       "output": [
           "Rédiger un pacte d'associés pour {target} en définissant clairement les droits et obligations des parties.",
           "Clarifier les modalités d'un pacte d'associés pour {target} avec une description détaillée de l'entrée d'un nouvel investisseur."
       ]
    },
    "transaction": {
       "input": [
           "Un contrat de transaction pour résoudre un litige sur {target} est nécessaire, mais c'est écrit de travers.",
           "Demande de transaction concernant {target} qui manque de clarté."
       ],
       "output": [
           "Formuler les termes d'une transaction pour résoudre un litige sur {target} en précisant les engagements de chaque partie.",
           "Clarifier une transaction relative à {target} avec une description détaillée des modalités de règlement."
       ]
    }
}

# Liste d'exemples de cibles (situation typiques en droit des affaires)
cibles = [
    "l'acquisition d'un portefeuille de brevets",
    "la vente d'un fonds de commerce de boulangerie",
    "la cession de parts dans une société de distribution alimentaire",
    "l'alliance stratégique entre deux entreprises de logistique",
    "la location d'un supermarché en centre-ville",
    "la sécurisation d'un paiement important dans le secteur de la construction",
    "le renouvellement d'un contrat de distribution de produits high-tech",
    "l'entrée d'un nouvel investisseur dans une start-up technologique",
    "la résolution d'un litige commercial entre partenaires"
]

# Générer 1000 exemples
exemples = []
num_examples = 1000

for _ in range(num_examples):
    categorie = random.choice(list(templates.keys()))
    cible = random.choice(cibles)
    input_template = random.choice(templates[categorie]["input"])
    output_template = random.choice(templates[categorie]["output"])
    exemple = {
        "input": input_template.format(target=cible),
        "output": {
            "type": categorie,
            "reformulated_query": output_template.format(target=cible)
        }
    }
    exemples.append(exemple)

# Enregistrer les exemples dans un fichier JSON
with open("synthetic_dataset.json", "w", encoding="utf-8") as f:
    json.dump(exemples, f, ensure_ascii=False, indent=2)

print("Dataset de 1000 exemples généré et sauvegardé dans 'synthetic_dataset.json'.")
