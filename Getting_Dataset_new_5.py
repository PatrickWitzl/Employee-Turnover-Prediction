import datetime
import pandas as pd
import random
from datetime import date, timedelta
from faker import Faker
import numpy as np
import time


# Start-Timerå
start_time = time.time()

# Fake-Generator initialisieren
random.seed(42)
Faker.seed(42)
np.random.seed(42)

# Fake-Generator initialisieren
fake = Faker()

# Variablen definieren
num_employees = 630
years_before_start = range(1980, 2010)
years = range(2010, 2025)
months = range(1, 13)  # Jeder Monat von Januar (1) bis Dezember (12)
positions = ["Bürokraft", "Fachkraft", "Leitung Büro", "Leitung Landeszentrale", "Abteilungsleiter",
             "Unterabteilungsleiter"]
salary_groups = {
    "Bürokraft": "E5",
    "Fachkraft": "E6",
    "Leitung Büro": "E8",
    "Leitung Landeszentrale": "E10",
    "Abteilungsleiter": "E12",
    "Unterabteilungsleiter": "E10"
}
genders = ["Männlich", "Weiblich"]
absence_reasons = ["Krankheit", "Elternzeit", "Sabbatical", "keine"]
locations = ["Büro", "Landeszentrale", "Bund"]
education_levels = ["Doktorrand", "Masterabschluss", "Bachelorabschluss", "Berufsausbildung"]
job_levels = ["Einstiegslevel", "Mittellevel", "Seniorlevel", "Executivelevel"]
work_models = ["Vollzeit", "Teilzeit", "Homeoffice"]

# Führungskräftepositionen und Nichtführungskräfte mischen
leadership_positions = (
        ["Leitung Landeszentrale"] * 4 +
        ["Leitung Büro"] * 40 +
        ["Abteilungsleiter"] * 10 +
        ["Unterabteilungsleiter"] * 10
)
non_leadership_positions = ["Bürokraft"] * 300 + ["Fachkraft"] * (num_employees - len(leadership_positions) - 300)
all_positions = leadership_positions + non_leadership_positions
random.shuffle(all_positions)

# Funktionsdefinitionen
def generiere_mitarbeiter_ids(num_employees):
    """Generiert eine Liste von zufälligen, einzigartigen Mitarbeiter-IDs."""
    if num_employees <= 0:
        raise ValueError("Die Anzahl der Mitarbeiter muss größer als 0 sein.")
    mitarbeiter_ids = set()
    while len(mitarbeiter_ids) < num_employees:
        mitarbeiter_ids.add(random.randint(10000, 99999))
    return list(mitarbeiter_ids)

def generate_employee_data(position, years, existing_hiring_date=None):
    """
    Logische Verknüpfung von Alter, Geburtsdatum und Einstellungsdatum.
    Mindestalter beträgt 25 Jahre.

    Parameter:
    - position: Position des Mitarbeiters.
    - years: Ein Bereich (range) oder eine Liste mit zulässigen Jahren für current_year.
    - existing_hiring_date: (Optional) Vorgegebenes Einstellungsdatum.

    Rückgabewerte:
    - Alter: Alter des Mitarbeiters.
    - Geburtsdatum: Berechnetes Geburtsdatum basierend auf dem Alter.
    - Einstellungsdatum: Entweder berechneter Wert oder übergebener Wert.
    """

    # Wähle ein zufälliges Jahr aus der übergebenen "years"-Variable
    current_year = random.choice(years)

    # Altersverteilungsdaten
    age_distribution = {
        "Bürokraft": [(20, 29, 0.3), (30, 39, 0.4), (40, 49, 0.2), (50, 65, 0.1)],
        "Fachkraft": [(25, 34, 0.2), (35, 44, 0.45), (45, 54, 0.25), (55, 65, 0.1)],
        "Leitung Büro": [(30, 39, 0.25), (40, 49, 0.4), (50, 59, 0.25), (60, 65, 0.1)],
        "Leitung Landeszentrale": [(35, 44, 0.2), (45, 54, 0.45), (55, 60, 0.25), (61, 65, 0.1)],
        "Abteilungsleiter": [(30, 39, 0.25), (40, 49, 0.4), (50, 59, 0.25), (60, 65, 0.1)],
        "Unterabteilungsleiter": [(30, 39, 0.3), (40, 49, 0.4), (50, 59, 0.2), (60, 65, 0.1)],
        "Standard": [(25, 34, 0.3), (35, 44, 0.4), (45, 54, 0.2), (55, 65, 0.1)],
    }

    # Altersverteilung anpassen
    age_ranges = age_distribution.get(position, age_distribution["Standard"])
    if current_year < 2015:
        for i in range(len(age_ranges)):
            min_age, max_age, probability = age_ranges[i]
            if min_age >= 60:  # Ruhestandsalter
                age_ranges[i] = (min_age, max_age, probability * 0.5)

    # Alter basierend auf der Altersverteilung wählen
    possible_ages = []
    probabilities = []
    for age_range in age_ranges:
        min_age, max_age, probability = age_range
        possible_ages.extend(range(min_age, max_age + 1))
        probabilities.extend([probability] * (max_age - min_age + 1))

    # Mindestalter 25
    age = max(random.choices(possible_ages, weights=probabilities, k=1)[0], 25)

    # Geburtsjahr und Geburtsdatum berechnen
    birth_year = current_year - age
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    birth_date = date(birth_year, birth_month, birth_day)

    # Mindestalter für den Ruhestand, basierend auf der Position
    position_retirement_ages = {
        "Abteilungsleiter": 62, "Leitung Landeszentrale": 65, "Standard": 60
    }
    min_retirement_age = position_retirement_ages.get(position, position_retirement_ages["Standard"])
    if age < min_retirement_age:
        age = min_retirement_age

    # Einstellungsdatum berechnen, falls nicht vorgegeben
    if existing_hiring_date is None:
        minimum_hiring_year = birth_year + 25
        maximum_hiring_year = min(current_year - 10, birth_year + 40)

        # Sicherstellen, dass der Bereich nicht leer ist
        if minimum_hiring_year > maximum_hiring_year:
            minimum_hiring_year, maximum_hiring_year = maximum_hiring_year, minimum_hiring_year

        hiring_year = random.randint(minimum_hiring_year, maximum_hiring_year)
        hiring_date = date(hiring_year, random.randint(1, 12), random.randint(1, 28))
    else:
        hiring_date = existing_hiring_date

    return {
        "Alter": age,
        "Geburtsdatum": birth_date,
        "Einstellungsdatum": hiring_date,
    }

def generate_valid_hiring_date():
    """Erzeugt ein gültiges Einstellungsdatum."""
    return fake.date_between(start_date='-30y', end_date='-1y')

def generate_satisfaction(flexibility, overwork, salary, benefits, illness_days, age, expected_salary=None):
    """
    Berechnet die Zufriedenheit eines Mitarbeiters basierend auf verschiedenen Faktoren.
    """
    base_satisfaction = 7  # Grundzufriedenheit
    flex_factor = (flexibility - 5) * 0.50
    overwork_factor = -0.5 * max(overwork - 5, 0) ** 1.2
    salary_factor = 0.2 * (salary / 1000)  # Gehalt wirkt positiv

    # Berücksichtigung der Gehaltserwartung
    if expected_salary:
        # Unterschied zwischen tatsächlichem und erwartetem Gehalt
        salary_gap = expected_salary - salary
        if salary_gap > 0:  # Wenn tatsächliches Gehalt unter der Erwartung liegt
            salary_factor -= 0.05 * salary_gap / 1000  # Negativer Einfluss abhängig von der Differenz

    benefits_factor = (benefits - 5) * 0.5
    illness_factor = -0.2 * illness_days ** 1.1
    age_factor = -0.3 if age > 60 else 0.2 if age < 30 else 0

    # Zufriedenheit dynamisch begrenzen
    return max(1, min(10,
                      base_satisfaction + flex_factor + overwork_factor + salary_factor +
                      benefits_factor + illness_factor + age_factor))

def calculate_expected_salary(position, age, education_level, work_model, base_salary=30000):
    """
       Berechnet die erwartete Gehaltshöhe eines Mitarbeiters basierend auf verschiedenen Parametern.

       :param position: Position des Mitarbeiters.
       :param age: Alter des Mitarbeiters.
       :param education_level: Bildungsniveau.
       :param work_model: Arbeitsmodell (z.B. Vollzeit oder Teilzeit).
       :param base_salary: Basisgehalt für Einstiegslevel-Positionen.
       :return: Erwartetes Gehalt in Euro.
       """
    # Grundgehalt je nach Position
    position_multiplier = {
        "Bürokraft": 1.0,
        "Fachkraft": 1.2,
        "Leitung Büro": 1.5,
        "Unterabteilungsleiter": 1.8,
        "Abteilungsleiter": 2.0,
        "Leitung Landeszentrale": 2.5
    }
    pos_factor = position_multiplier.get(position, 1.0)

    # Alterszuschlag (je 5 Jahre 5 % auf das Grundgehalt)
    age_factor = 1 + (max(age - 25, 0) // 5) * 0.05

    # Bildungszuschlag
    education_multiplier = {
        "Berufsausbildung": 1.0,
        "Bachelorabschluss": 1.1,
        "Masterabschluss": 1.2,
        "Doktorrand": 1.3
    }
    edu_factor = education_multiplier.get(education_level, 1.0)

    # Arbeitsmodell (z.B. Teilzeit hat geringere Erwartungen)
    if work_model == "Teilzeit":
        work_model_factor = 0.7
    elif work_model == "Homeoffice":
        work_model_factor = 0.9
    else:
        work_model_factor = 1.0  # Vollzeit

    # Berechnung des erwarteten Gehalts:
    expected_salary = base_salary * pos_factor * age_factor * edu_factor * work_model_factor

    return expected_salary

def adjust_for_inflation(base_salary, current_year, hire_year):
    """
       Anpassung des Gehalts durch jährliche Inflation (z.B. 2 % pro Jahr).
       """
    inflation_rate = 0.02
    years_passed = max(current_year - hire_year, 0)
    return base_salary * ((1 + inflation_rate) ** years_passed)

def generate_illness_days(satisfaction, age, work_model, leadership_stress=False, long_term_illness_prob=0.1):
    if random.random() < long_term_illness_prob:  # Langfristige Erkrankungen
        illness_days = max(
            15,
            int(np.clip(np.random.normal(loc=25 - satisfaction, scale=8), 15, 50))  # Skala erhöht (max. 50)
        )
    else:
        # Verstärkte Faktoren
        base_sickness = (5 + age // 8) / 12  # Altersquote stärker gewichtet
        satisfaction_factor = ((10 - satisfaction) ** 2) * 0.5  # Niedrige Zufriedenheit stärker negativ
        flexibility_factor = -3 if work_model == "Teilzeit" else 0  # Teilzeit stärker positiv
        age_factor = 4 if age > 55 else -2 if age < 25 else 0  # Alterseffekte verstärkt
        leadership_factor = 5 if leadership_stress else 0  # Stress durch Führungsverantwortung höher
        illness_days = int(np.clip(
            base_sickness + satisfaction_factor + flexibility_factor + age_factor + leadership_factor,
            0,  # Keine negativen Werte
            40  # Obergrenze
        ))

    illness_days = max(0, illness_days)

    return illness_days

def entscheiden_status(employee, satisfaction, overwork, performance_score,
                       salary, illness_days, tenure, training_costs, team_size, subordinates,
                       workplace_flexibility, job_role_progression, job_level,
                       switching_readiness):
    """
    Entscheidet, ob ein Mitarbeiter aktiv bleibt, in den Ruhestand geht oder ausscheidet.
    """
    age = employee["Alter"]

    termination_probability = 0.1  # Ausgangswahrscheinlichkeit für Kündigung
    retirement_probability = 0.0  # Ruhestandswahrscheinlichkeit

    # Altersabhängige Wahrscheinlichkeiten für Ruhestand
    if 60 <= age <= 65:
        retirement_probability += 0.3
        if satisfaction > 6:
            retirement_probability -= 0.1
        if job_level >= 2:
            retirement_probability -= 0.05
        if performance_score >= 4:
            retirement_probability -= 0.1
    elif age > 65:
        retirement_probability += 0.5

    # Zufriedenheit
    if satisfaction <= 2:
        termination_probability += 0.45
    elif 2 < satisfaction <= 5:
        termination_probability += 0.2
    elif satisfaction >= 6:
        termination_probability -= (satisfaction - 5) * 0.08

    # Überstunden
    if overwork > 15:
        termination_probability += 0.25 + 0.01 * (overwork - 15)
    elif overwork < 5:
        termination_probability -= 0.1

    # Gehalt
    expected_salary = 40000 + (age - 25) * 600
    if salary < expected_salary * 0.7:
        termination_probability += 0.3
    elif salary >= expected_salary * 1.3:
        termination_probability -= 0.1

    # Krankheitsausfälle
    if illness_days > 20:
        termination_probability += 0.15
    elif illness_days < 5:
        termination_probability -= 0.05

    # Weiterbildungsinvestitionen
    if training_costs > 2000:
        termination_probability -= 0.05

    # Teamgröße und Untergebene
    if team_size > 15 and employee["Position"] in ["Abteilungsleiter", "Unterabteilungsleiter", "Leitung Büro"]:
        termination_probability += 0.1
    if subordinates > 5:
        termination_probability += 0.05

    # Karriereentwicklung
    if job_role_progression <= 3:
        termination_probability += 0.2
    elif job_role_progression >= 7:
        termination_probability -= 0.1

    # Wechselbereitschaft
    if switching_readiness > 0.7:
        termination_probability += 0.2
    elif 0.4 <= switching_readiness <= 0.7:
        termination_probability += 0.1
    else:
        termination_probability -= 0.1

    # Kombinierte Wahrscheinlichkeiten
    combined_probability = retirement_probability + termination_probability
    random_roll = random.uniform(0, 1)

    # Entscheiden
    if random_roll < retirement_probability:
        return "Ruhestand"
    elif random_roll < combined_probability:
        return "Ausgeschieden"
    else:
        return "Aktiv"

def calculate_switching_readiness_with_emotions(satisfaction, overwork, salary, expected_salary, workplace_flexibility,
                                                illness_days, emotional_factors):
    """
    Berechnet eine schwächere Wechselbereitschaft basierend auf objektiven Faktoren und einem emotionalen Profil.
    - satisfaction: Zufriedenheit (1 bis 10)
    - overwork: Anzahl der Überstunden
    - salary: Tatsächliches Gehalt
    - expected_salary: Erwartetes Gehalt
    - workplace_flexibility: Arbeitsplatzflexibilität (1 bis 10)
    - illness_days: Anzahl der Krankheitstage
    - emotional_factors: Emotionale Werte (z. B. Stress, Konflikte, Anerkennung, etc.)

    Rückgabe:
        readiness: Wechselbereitschaft, ein Wert zwischen 0 und 1.
    """
    # Ausgangswert für die Wechselbereitschaft
    readiness = 0.3  # Leichter reduziert von vorher 0.5

    # Zufriedenheit und Überstunden einbauen
    if satisfaction <= 4:
        readiness += (5 - satisfaction) * 0.08  # Gewichtung etwas reduziert
    if overwork > 10:
        readiness += (overwork - 10) * 0.005  # Überstunden erhöhen Wechselbereitschaft schwächer

    # Gehalt
    if salary < expected_salary * 0.8:
        readiness += 0.15  # Reduzierung der Gewichtung von 0.2
    elif salary >= expected_salary * 1.2:
        readiness -= 0.12  # Positive Gehaltsanreize stärker berücksichtigt

    # Arbeitsplatzflexibilität
    if workplace_flexibility >= 7:
        readiness -= (workplace_flexibility - 6) * 0.06  # Positive Wirkung verstärkt
    elif workplace_flexibility <= 3:
        readiness += (4 - workplace_flexibility) * 0.03  # Negative Wirkung reduziert

    # Krankheitstage
    if illness_days > 10:
        readiness += (illness_days - 10) * 0.005  # Gewichtung halbiert

    # Emotionale Faktoren hinzufügen
    stress = emotional_factors.get("stress", 5)
    recognition = emotional_factors.get("recognition", 5)
    work_environment = emotional_factors.get("work_environment", 5)
    future_opportunities = emotional_factors.get("future_opportunities", 5)
    team_conflicts = emotional_factors.get("team_conflicts", 5)
    boredom = emotional_factors.get("boredom", 5)

    # Stress erhöht die Wechselbereitschaft (leicht abgeschwächt)
    readiness += (stress - 5) * 0.03

    # Anerkennung reduziert Wechselbereitschaft (leicht verstärkt)
    readiness -= (recognition - 5) * 0.05

    # Positives Arbeitsumfeld reduziert Wechselbereitschaft (leicht verstärkt)
    readiness -= (work_environment - 5) * 0.04

    # Zukunftsperspektiven stärker gewichtet
    readiness -= (future_opportunities - 5) * 0.05

    # Konflikte erhöhen die Wechselbereitschaft (leicht abgeschwächt)
    readiness += (team_conflicts - 5) * 0.04

    # Langeweile (Boreout) erhöht ebenfalls die Wechselbereitschaft (leicht abgeschwächt)
    readiness += (boredom - 5) * 0.02

    # Grenzen sicherstellen
    readiness = max(0, min(1, readiness))

    return round(readiness, 2)

# Schritt 1: Fixierte Mitarbeiterinformationen erstellen
mitarbeiter_ids = generiere_mitarbeiter_ids(num_employees)  # Mitarbeiter-IDs erstellen
persistent_employee_data = []  # Temporäre Liste für alle Mitarbeiterdaten

for i in range(num_employees):
    position = all_positions[i]  # Position aus der zufälligen Liste auswählen

    # Mitarbeiterdaten (Alter und Einstellungsdatum) generieren
    employee_data = generate_employee_data(position, years_before_start)  # Daten generieren
    age = employee_data["Alter"]
    geburtsdatum = employee_data["Geburtsdatum"]
    einstellungsdatum = employee_data["Einstellungsdatum"]  # Sicherstellen, dass Einstellungsdatum korrekt gesetzt ist

    # Konsistente einmalige Berechnungen für Bildung und Arbeitsmodell
    education_level = random.choice(education_levels)  # Zufällige Auswahl der Bildung
    work_model = random.choice(work_models)  # Arbeitszeitmodell wählen (z. B. Vollzeit, Teilzeit)

    # Erwartetes Gehalt berechnen (konsistent)
    expected_salary = calculate_expected_salary(
        position=position,
        age=age,
        education_level=education_level,
        work_model=work_model
    )

    # Mitarbeiterdatensatz erstellen
    mitarbeiter = {
        "Mitarbeiter_ID": mitarbeiter_ids[i],
        "Name": fake.name(),
        "Geschlecht": random.choice(genders),
        "Einstellungsdatum": einstellungsdatum,  # Übergebenes oder generiertes Datum
        "Geburtsdatum": geburtsdatum,
        "Position": position,
        "Education Level": education_level,
        "Arbeitszeitmodell": work_model,
        "Status": "Aktiv",  # Initial immer Aktiv
        "Fehlzeiten_Krankheitstage": generate_illness_days(
            satisfaction=random.randint(1, 10),  # Zufällige Zufriedenheit
            age=age,
            work_model=work_model,
            leadership_stress=(position in leadership_positions)  # Führungsposition?
        ),
        "Zufriedenheit": generate_satisfaction(
            flexibility=random.randint(0, 10),
            overwork=max(0, random.randint(0, 20)),
            salary=expected_salary,  # Konsistentes Gehalt verwenden
            benefits=random.randint(0, 10),
            illness_days=0,  # Initial ohne Krankheitstage
            age=age,
            expected_salary=expected_salary
        ),
        "Alter": age,
        "Wiedereinstellung_gesperrt": False,  # Zu Start immer False
    }

    # Mitarbeiterdaten hinzufügen
    persistent_employee_data.append(mitarbeiter)

# Übergabe der generierten Mitarbeiterdaten
print(f"{num_employees} Mitarbeiter erfolgreich initialisiert!")

# Schritt 2: Nachfolgeplanung initialisieren
nachfolgeplanung = []

# Schritt 2: Startphase der Zeitanalyse einleiten
# Simulation startet mit festgelegten Mitarbeitern
for year in years:
    for month in months:
        aktive_mitarbeiter = [
            e for e in persistent_employee_data if e["Status"] == "Aktiv"
        ]

        print(f"INFO: Jahr {year}, Monat {month}, Aktive Mitarbeiter: {len(aktive_mitarbeiter)}")

        # Hier können weitere vorbereitende Berechnungen oder Logiken eingefügt werden,
        # bevor neue Einstellungen oder geplante Nachfolger verarbeitet werden.

# Schritt 3: Monatlich variierende Daten generieren
data = []
for year in years:
    for month in months:
        neueinstellungen = []

        for eintrag in list(nachfolgeplanung):  # Kopie der Nachfolgeplanung, um Änderungen zu ermöglichen
            if eintrag["Jahr"] == year and eintrag["Monat"] == month:
                neuer_mitarbeiter_id = random.randint(10000, 99999)
                while neuer_mitarbeiter_id in mitarbeiter_ids:
                    neuer_mitarbeiter_id = random.randint(10000, 99999)
                mitarbeiter_ids.append(neuer_mitarbeiter_id)

                # Generiere Mitarbeiterdaten, einschließlich Alter
                position = eintrag["Position"]
                employee_data = generate_employee_data(position, years)

                # Alter und Geburtsdatum direkt aus den generierten Daten verwenden
                age = employee_data["Alter"]
                geburtsdatum = employee_data["Geburtsdatum"]

                # NEU: Erwartetes Gehalt berechnen
                expected_salary = calculate_expected_salary(
                    position=position,
                    age=age,
                    education_level=random.choice(education_levels),  # Zufällige Auswahl der Bildung
                    work_model=random.choice(work_models)  # Zufälliges Arbeitszeitmodell
                )

                # Krankheitstage berechnen
                illness_days_calculated = generate_illness_days(
                    satisfaction=random.randint(1, 10),
                    age=age,
                    work_model=random.choice(work_models) ,
                    leadership_stress=(position in leadership_positions)
                )

                # Neuer Mitarbeiterdatensatz
                neuer_mitarbeiter = {
                    "Mitarbeiter_ID": neuer_mitarbeiter_id,
                    "Name": fake.name(),
                    "Geschlecht": random.choice(["männlich", "weiblich", "divers"]),
                    "Einstellungsdatum": employee_data["Einstellungsdatum"],
                    "Geburtsdatum": geburtsdatum,
                    "Alter": age,
                    "Position": position,
                    "Status": "Aktiv",

                    # Krankheitstage speichern
                    "Fehlzeiten_Krankheitstage": illness_days_calculated,
                    "Zufriedenheit": generate_satisfaction(
                        flexibility=random.randint(0, 10),
                        overwork=max(0, random.randint(0, 20)),  # Vorübergehender Wert
                        salary=random.randint(25000, 80000),  # Tatsächliches Gehalt
                        benefits=random.randint(0, 10),
                        illness_days=illness_days_calculated,  # Fix: Referenziere direkt die berechneten Krankheitstage
                        age=age,
                        expected_salary=expected_salary  # Erwartetes Gehalt einfügen
                    ),
                    "Wiedereinstellung_gesperrt": False,
                }

                # Neueinstellungen erstellen und Nachfolgeplanung aktualisieren
                neueinstellungen.append(neuer_mitarbeiter)
                nachfolgeplanung.remove(eintrag)

        # Alte geplante Einträge entfernen
        nachfolgeplanung = [
            entry for entry in nachfolgeplanung if
            entry["Jahr"] > year or (entry["Jahr"] == year and entry["Monat"] > month)
        ]

        # Berücksichtige auch geplante Nachfolger aus der Nachfolgeplanung
        geplante_mitarbeiter = sum(
            1 for eintrag in nachfolgeplanung if eintrag["Jahr"] == year and eintrag["Monat"] >= month
        )
        aktive_mitarbeiter = sum(1 for employee in persistent_employee_data if employee["Status"] == "Aktiv")
        gesamte_mitarbeiter = aktive_mitarbeiter + geplante_mitarbeiter

        # Überprüfung von Abweichungen der Mitarbeiteranzahl
        abweichung = num_employees - aktive_mitarbeiter
        if abs(abweichung) > 33:
            zu_hinzufuegende_mitarbeiter = random.randint(8, 19)

            # Zusätzliche Mitarbeiter hinzufügen
            for _ in range(zu_hinzufuegende_mitarbeiter):
                neuer_mitarbeiter_id = random.randint(10000, 99999)
                while neuer_mitarbeiter_id in mitarbeiter_ids:
                    neuer_mitarbeiter_id = random.randint(10000, 99999)
                mitarbeiter_ids.append(neuer_mitarbeiter_id)

                # Position und Daten generieren
                position = random.choice(positions)
                employee_data = generate_employee_data(position, years)
                age = employee_data["Alter"]
                geburtsdatum = employee_data["Geburtsdatum"]  # Geburtsdatum direkt übernehmen

                # NEU: Erwartetes Gehalt berechnen
                expected_salary = calculate_expected_salary(
                    position=position,
                    age=age,
                    education_level=random.choice(education_levels),  # Zufällige Auswahl der Bildung
                    work_model=random.choice(work_models)  # Zufälliges Arbeitszeitmodell
                )

                illness_days = generate_illness_days(
                    satisfaction=random.randint(1, 10),  # Zufällige Zufriedenheit
                    age=age,
                    work_model=random.choice(work_models),  # Zufälliges Arbeitszeitmodell
                    leadership_stress=(position in leadership_positions)  # Führungsposition = mehr Stress
                )

                # Neuer Mitarbeiterdatensatz
                neuer_mitarbeiter = {
                    "Mitarbeiter_ID": neuer_mitarbeiter_id,
                    "Name": fake.name(),
                    "Geschlecht": random.choice(genders),
                    "Alter": age,
                    "Einstellungsdatum": employee_data["Einstellungsdatum"],
                    "Geburtsdatum": geburtsdatum,
                    "Position": position,
                    "Education Level": random.choice(education_levels),
                    "Arbeitszeitmodell": random.choice(work_models),
                    "Status": "Aktiv",

                    # Berechnete Krankheitstage speichern
                    "Fehlzeiten_Krankheitstage": illness_days,

                    # Zufriedenheit berechnen, basierend auf realistischen Werten
                    "Zufriedenheit": generate_satisfaction(
                        flexibility=random.randint(0, 10),
                        overwork=max(0, random.randint(0, 20)),  # Vorübergehender Wert
                        salary=random.randint(25000, 80000),  # Tatsächliches Gehalt
                        benefits=random.randint(0, 10),
                        illness_days=illness_days,  # Verweis auf berechnete Krankheitstage
                        age=age,
                        expected_salary=expected_salary  # Erwartetes Gehalt einfügen
                    ),

                    "Wiedereinstellung_gesperrt": False,

                    # Neue Felder für erweiterte Logik
                    "Workplace Flexibility": random.randint(0, 10),  # Flexibilität (0–10)
                    "Job Role Progression": random.randint(0, 10),  # Karriereentwicklung (0–10)
                    "Job Level": random.choice(list(range(1, 5))),  # Berufliches Level (1–4)
                    "Fortbildungskosten": random.randint(0, 5000) if random.random() < 0.5 else 0,  # Optional
                    "Team Size": random.randint(5, 25),  # Teamgröße (basierend auf Position)
                    "Anzahl Untergebene": max(0, positions.index(position) - 2)
                    if position in leadership_positions else 0  # Abhängig von Position
                }

                # Mitarbeiterdaten dauerhaft speichern
                persistent_employee_data.append(neuer_mitarbeiter)

        # Zusammenfassung für den Monat ausgeben
        print(
            f"INFO: {year}-{month:02d} abgeschlossen -> Aktive: {aktive_mitarbeiter}, "
            f"Geplante: {geplante_mitarbeiter}, Gesamt: {gesamte_mitarbeiter}, "
            f"Neueinstellungen: {len(neueinstellungen)}"
        )

        for employee in persistent_employee_data:
            # Überspringe Mitarbeiter, die ausgeschieden oder im Ruhestand sind
            if employee["Status"] in ["Ruhestand", "Ausgeschieden"]:
                continue  # Mitarbeiter überspringen, wenn nicht aktiv

            # Aktuelles Datum und Alter berechnen
            current_date = datetime.datetime(year, month, 1).date()
            employee["Alter"] = (current_date.year - employee["Geburtsdatum"].year) - (
                    (current_date.month, current_date.day) < (
                employee["Geburtsdatum"].month, employee["Geburtsdatum"].day)
            )

            # **Überprüfung: Keine Daten vor dem Einstellungsdatum**
            if current_date < employee["Einstellungsdatum"]:
                continue  # Überspringe diesen Mitarbeiter für Monate vor dem Einstellungsdatum

            # Relevante Werte berechnen
            satisfaction = employee["Zufriedenheit"]  # Zufriedenheitswert
            overwork = employee.get("Überstunden", random.randint(0, 20))  # Dynamische Überstunden
            salary = employee.get("Gehalt", random.randint(25000, 80000))  # Gehalt
            expected_salary = calculate_expected_salary(
                position=employee["Position"],
                age=employee["Alter"],
                education_level=employee["Education Level"],
                work_model=employee["Arbeitszeitmodell"]
            )
            workplace_flexibility = {
                "Vollzeit": random.randint(3, 6),
                "Teilzeit": random.randint(6, 9),
                "Homeoffice": random.randint(7, 10)
            }.get(employee["Arbeitszeitmodell"], random.randint(1, 5))  # Standardwert, falls nicht definiert

            # Krankheitstage aktualisieren
            leadership_stress = employee["Position"] in leadership_positions
            employee["Fehlzeiten_Krankheitstage"] = generate_illness_days(
                satisfaction=satisfaction,
                age=employee["Alter"],
                work_model=employee["Arbeitszeitmodell"],
                leadership_stress=leadership_stress
            )

            # Emotionale Faktoren initialisieren (falls nicht vorhanden)
            emotional_factors = employee.get("Emotional Factors", {
                "stress": random.randint(1, 10),
                "recognition": random.randint(1, 10),
                "work_environment": random.randint(1, 10),
                "future_opportunities": random.randint(1, 10),
                "team_conflicts": random.randint(1, 10),
                "boredom": random.randint(1, 10),
            })

            # Wechselbereitschaft berechnen
            switching_readiness = calculate_switching_readiness_with_emotions(
                satisfaction=satisfaction,
                overwork=overwork,
                salary=salary,
                expected_salary=expected_salary,
                workplace_flexibility=workplace_flexibility,
                illness_days=employee["Fehlzeiten_Krankheitstage"],
                emotional_factors=emotional_factors
            )

            # Standardwert für Status setzen
            neuer_status = "Aktiv"  # Standardmäßig "Aktiv"

            # Ruhestand ab 67 Jahren
            if employee["Alter"] >= 67:
                employee["Status"] = "Ruhestand"
                employee["Austrittsdatum"] = current_date
                continue

            # Ruhestandswahrscheinlichkeit (zwischen 60 und 69 Jahren)
            elif 60 <= employee["Alter"] < 70:
                ruhestand_wahrscheinlichkeit = max(0.05, 0.05 * (5 - satisfaction))  # Basis 5% Wahrscheinlichkeit
                if random.random() < ruhestand_wahrscheinlichkeit:
                    neuer_status = "Ruhestand"

            # Kündigungslogik basierend auf Wechselbereitschaft und schlechter Zufriedenheit
            elif switching_readiness > 0.7 and satisfaction < 4:
                neuer_status = "Ausgeschieden"

            # Kündigungswahrscheinlichkeit bei langjährigen unzufriedenen Mitarbeitern
            elif employee["Einstellungsdatum"].year < year - 5 and satisfaction < 5 and random.random() < 0.1:
                neuer_status = "Ausgeschieden"

            # Aktualisiere den Status des Mitarbeiters
            employee["Status"] = neuer_status

            # Sicherstellen, dass Ruhestandsregelungen ab 67 Jahren korrekt eingehalten werden
            if employee["Alter"] >= 67 and employee["Status"] != "Ruhestand":
                print(f"DEBUG: Fehlerhafte Statusaktualisierung für Mitarbeiter {employee['Mitarbeiter_ID']}.")
                employee["Status"] = "Ruhestand"

            # Austrittsdatum und Nachfolgeregelung bei "Ausgeschieden" oder "Ruhestand"
            if employee["Status"] in ["Ausgeschieden", "Ruhestand"]:
                employee["Austrittsdatum"] = current_date
                employee["Austrittsjahr"] = year
                employee["Austrittsmonat"] = month
                employee["Letzter_Aktivmonat"] = f"{year:04d}-{month:02d}"

                # NEUER BLOCK: Direkte Nachfolgeplanung für diesen Mitarbeiter
                nachfolgeplanung.append({
                    "Jahr": year,
                    "Monat": month + 1 if month < 12 else 1,
                    "Position": employee["Position"]
                })

                # Planen von Nachfolgern, falls es notwendig ist
                if abweichung > 0:
                    einstellungsmonat = month + 1
                    einstellungsjahr = year
                    if einstellungsmonat > 12:
                        einstellungsmonat -= 12
                        einstellungsjahr += 1
                    for _ in range(min(abweichung, 15)):  # Maximal 15 Nachfolger planen
                        position = random.choice(positions)
                        geschlecht = random.choice(genders)
                        nachfolgeplanung.append({
                            "Jahr": einstellungsjahr,
                            "Monat": einstellungsmonat,
                            "Position": position,
                            "Geschlecht": geschlecht,
                        })
                        print(
                            f"DEBUG: Nachfolger geplant. Position: {position}, Einstellung: {einstellungsjahr}-{einstellungsmonat:02d}")

            # Verstärkte Korrelationen für Krankheitstage
            current_date = datetime.datetime(year, month, 1).date()
            geburtsdatum = employee["Geburtsdatum"]
            age = (current_date.year - geburtsdatum.year) - (
                    (current_date.month, current_date.day) < (geburtsdatum.month, geburtsdatum.day)
            )

            leadership_stress = employee["Position"] in leadership_positions

            # Berechnung der Krankheitstage über die Funktion
            illness_days = generate_illness_days(
                satisfaction=employee.get("Zufriedenheit", 7),
                age=age,
                work_model=employee.get("Arbeitszeitmodell", "Vollzeit"),
                leadership_stress=leadership_stress
            )

            # Abwesenheitsgrund dynamisch wählen
            if illness_days > 0:
                if illness_days > 20:  # Längere Krankheitstage
                    abwesenheitsgrund = "Erkrankung mit Reha"
                else:
                    abwesenheitsgrund = random.choices(
                        absence_reasons,
                        weights=[50 - age // 10, 30 + (age // 20), 10, 10],  # Gewichtung basierend auf Alter
                        k=1
                    )[0]
            else:
                abwesenheitsgrund = "keine"

            # Gehalt anhand einer unregelmäßigen Verteilung und Zufriedenheit berechnen
            base_salary = (
                random.randint(30000, 45000) if employee["Position"] in ["Bürokraft", "Fachkraft"]
                else random.randint(50000, 80000)
            )

            # Gehaltsanpassung: Unregelmäßig und zufallsbasiert
            irregularity_factor = random.triangular(0.8, 1.2, 1.5)  # Asymmetrische Verteilung
            inflation_adjustment = base_salary * ((year - 2015) * 0.02)  # 2% Inflation pro Jahr
            satisfaction_bonus = employee.get("Zufriedenheit", 5) * 500  # Zufriedenheit beeinflusst Gehalt

            gehalt = (base_salary * irregularity_factor) + inflation_adjustment + satisfaction_bonus

            base_data = {
                "Jahr": year,
                "Monat": month,
                "Mitarbeiter_ID": employee["Mitarbeiter_ID"],
                "Name": employee["Name"],
                "Geschlecht": employee["Geschlecht"],
                "Einstellungsdatum": employee["Einstellungsdatum"],
                "Austrittsdatum": employee.get("Austrittsdatum", None),
                "Position": employee["Position"],
                "Education Level": employee["Education Level"],
                "Alter": age,
                "Geburtsdatum": employee["Geburtsdatum"],
                "Gehaltsgruppe": salary_groups[employee["Position"]],
                "Grundausbildung": random.choice(["Ja", "Nein"]),
                "Interne Weiterbildungen": random.randint(0, 5),
                "Planstelle": random.choice(["Ja", "Nein"]),
                "Zeit bis zur Rente": max(0, 67 - age),
                "Verheiratet": random.choice(["Ja", "Nein"]),
                "Kinder": random.choice(["Ja", "Nein"]),
                "Urlaubstage_genommen": random.randint(0, 30 // 12),
                "Überstunden": max(0,
                   int(np.random.normal(
                       loc=(15 + positions.index(employee["Position"]) * 5),
                       scale=5  # Größere Streuung für realistischere Verteilung
                   ))),
                "Fehlzeiten_Krankheitstage": illness_days,
                "Abwesenheitsgrund": abwesenheitsgrund,
                "Jährliche Leistungsbewertung": random.randint(1, 5),
                "Standort": random.choices(locations, weights=[70, 20, 10])[0],
                "Wechselbereitschaft": employee.get("Wechselbereitschaft", round(random.uniform(0, 1), 2)),
                "Fortbildungskosten": employee.get("Fortbildungskosten",
                                                   random.randint(0, 5000) if random.random() < 0.5 else 0),
                "Anzahl Untergebene": employee.get("Anzahl Untergebene",
                                                   max(0, positions.index(employee["Position"]) - 2)
                                                   if employee["Position"] in leadership_positions else 0),
                "Job Role Progression": employee.get("Job Role Progression", random.choices(
                    ["Beförderung", "Seitlicher Wechsel", "Verlassen und Wiedereinstieg", "Keine Veränderung"],
                    weights=[30, 20, 10, 40])[0]),
                "Gehalt": gehalt,
                "Workplace Flexibility": employee.get("Workplace Flexibility", random.randint(0, 10)),
                "Team Size": employee.get("Team Size", random.randint(5, 25)),
                "Coaching": random.choice(["Ja", "Nein"]),
                "Job Level": employee.get("Job Level", random.choice(job_levels)),
                "Arbeitszeitmodell": employee["Arbeitszeitmodell"],
                "Zufriedenheit": employee["Zufriedenheit"],
                "Status": employee["Status"],
                "Tenure": year - employee["Einstellungsdatum"].year
            }

            # Kinder, Grundausbildung und Status stabil halten
            if year > min(years) or month > min(months):  # Für alle Zeiten außer dem ersten Monat/Jahr prüfen
                # Vorherige Datensätze überprüfen
                prev_entry = next(
                    (entry for entry in data if
                     entry.get("Jahr") == year and
                     entry.get("Monat") == month - 1 and
                     entry.get("Mitarbeiter_ID") == employee.get("Mitarbeiter_ID")),
                    None
                )

                if not prev_entry:  # Falls vorheriger Monat nicht existiert, prüfe auf Vorjahr-Dezember
                    prev_entry = next(
                        (entry for entry in data if
                         entry.get("Jahr") == year and
                         entry.get("Monat") == month - 1 and
                         entry.get("Mitarbeiter_ID") == employee.get("Mitarbeiter_ID")),
                        None
                    )
                if prev_entry:
                    if prev_entry["Kinder"] == "Ja":
                        base_data["Kinder"] = "Ja"
                    if prev_entry["Grundausbildung"] == "Ja":
                        base_data["Grundausbildung"] = "Ja"

            data.append(base_data)

# Schritt 3: DataFrame erstellen
df_updated = pd.DataFrame(data)

# Alle Einstellungsdaten validieren und sicherstellen, dass nur das Datum ohne Zeit gespeichert wird
df_updated['Einstellungsdatum'] = pd.to_datetime(df_updated['Einstellungsdatum'], errors='coerce', format='%Y-%m-%d')
df_updated['Austrittsdatum'] = pd.to_datetime(df_updated['Austrittsdatum'], errors='coerce', format='%Y-%m-%d')
df_updated['Geburtsdatum'] = pd.to_datetime(df_updated['Geburtsdatum'], errors='coerce', format='%Y-%m-%d')
# Daten speichern, ohne Zeitangaben
df_updated.to_csv("HR_Testdatensatz.csv", index=False, date_format="%Y-%m-%d")

end_time = time.time()
print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")


