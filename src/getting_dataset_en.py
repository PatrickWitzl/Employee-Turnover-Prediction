import datetime
import pandas as pd
import random
from datetime import date, timedelta
from faker import Faker
import numpy as np
import time

# Start timer
start_time = time.time()

# Initialize Faker generator
random.seed(42)
Faker.seed(42)
np.random.seed(42)

# Initialize Faker generator
fake = Faker()

# Define variables
num_employees = 630
start_year = 2010  # Year when simulation starts
start_month = 1  # Month when simulation starts
years = range(2010, 2025)
months = range(1, 13)  # Every month from January (1) to December (12)
positions = ["Office Clerk", "Specialist", "Office Manager", "Headquarters Manager", "Department Head",
             "Assistant Department Head"]
salary_groups = {
    "Office Clerk": "E5",
    "Specialist": "E6",
    "Office Manager": "E8",
    "Headquarters Manager": "E10",
    "Department Head": "E12",
    "Assistant Department Head": "E10"
}
genders = ["Male", "Female"]
absence_reasons = ["Illness", "Parental Leave", "Sabbatical", "None"]
locations = ["Office", "Headquarters", "Federal"]
education_levels = ["Doctoral Candidate", "Master Degree", "Bachelor Degree", "Vocational Training"]
job_levels = ["Entry Level", "Mid Level", "Senior Level", "Executive Level"]
work_models = ["Full-time", "Part-time", "Homeoffice"]

# Mix leadership and non-leadership positions
leadership_positions = (
        ["Headquarters Manager"] * 4 +
        ["Office Manager"] * 40 +
        ["Department Head"] * 10 +
        ["Assistant Department Head"] * 10
)
non_leadership_positions = ["Office Clerk"] * 300 + ["Specialist"] * (num_employees - len(leadership_positions) - 300)
all_positions = leadership_positions + non_leadership_positions
random.shuffle(all_positions)


# Function definitions

def generate_employee_ids(num_employees):
    """Generates a list of random, unique employee IDs."""
    if num_employees <= 0:
        raise ValueError("The number of employees must be greater than 0.")
    employee_ids = set()
    while len(employee_ids) < num_employees:
        employee_ids.add(random.randint(10000, 99999))
    return list(employee_ids)


def generate_employee_data(position, current_year, current_month):
    """
    Generates employee data with appropriate hiring dates matching the current month and year.

    Parameters:
    - position: Employee's position.
    - current_year: Current simulation year (int).
    - current_month: Current simulation month (int).

    Returns:
    - Age: Employee's age.
    - Birthdate: Calculated birthdate based on age.
    - Hiring date: Fixed to the current simulation month and year.
    """
    # Age distribution data
    age_distribution = {
        "Office Clerk": [(20, 29, 0.3), (30, 39, 0.4), (40, 49, 0.2), (50, 65, 0.1)],
        "Specialist": [(25, 34, 0.2), (35, 44, 0.45), (45, 54, 0.25), (55, 65, 0.1)],
        "Office Manager": [(30, 39, 0.25), (40, 49, 0.4), (50, 59, 0.25), (60, 65, 0.1)],
        "Headquarters Manager": [(35, 44, 0.2), (45, 54, 0.45), (55, 60, 0.25), (61, 65, 0.1)],
        "Department Head": [(30, 39, 0.25), (40, 49, 0.4), (50, 59, 0.25), (60, 65, 0.1)],
        "Assistant Department Head": [(30, 39, 0.3), (40, 49, 0.4), (50, 59, 0.2), (60, 65, 0.1)],
        "Standard": [(25, 34, 0.3), (35, 44, 0.4), (45, 54, 0.2), (55, 65, 0.1)],
    }

    age_ranges = age_distribution.get(position, age_distribution["Standard"])

    # Select age based on distribution
    possible_ages, probabilities = [], []
    for min_age, max_age, probability in age_ranges:
        possible_ages.extend(range(min_age, max_age + 1))
        probabilities.extend([probability] * (max_age - min_age + 1))
    age = max(random.choices(possible_ages, weights=probabilities, k=1)[0], 25)

    # Calculate birthdate
    birth_year = current_year - age
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    birth_date = date(birth_year, birth_month, birth_day)

    min_valid_hiring_date = birth_date + timedelta(days=25 * 365)

    # Hiring date fixed to 1st day of the month
    hiring_date = date(current_year, current_month, 1)

    # Ensure hiring date meets minimum date criterion
    if hiring_date < min_valid_hiring_date:
        hiring_date = min_valid_hiring_date

    return {
        "Age": age,
        "Birthdate": birth_date,
        "Hiring Date": hiring_date,
    }


def generate_valid_hiring_date(birth_date, year, month, min_age=25):
    """
    Generates a valid hiring date matching the given year and month and checks minimum age constraint.

    :param birth_date: Employee's birthdate (datetime.date)
    :param year: Desired hiring year (int)
    :param month: Desired hiring month (int)
    :param min_age: Minimum required age for employee (default: 25)
    :return: A datetime.date object representing a valid hiring date.
    """

    min_hiring_year = birth_date.year + min_age

    if year < min_hiring_year:
        raise ValueError(f"Cannot hire: Year {year} is before minimum hiring year {min_hiring_year}.")

    if month not in range(1, 13):
        raise ValueError(f"Invalid month: {month}. Month must be between 1 and 12.")

    if month == 2:
        day = random.randint(1, 28)
    elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
    else:
        day = random.randint(1, 31)

    hiring_date = date(year, month, day)

    min_valid_date = date(min_hiring_year, birth_date.month, birth_date.day)
    if hiring_date < min_valid_date:
        raise ValueError(f"Hiring date {hiring_date} is before reaching minimum age ({min_valid_date}).")

    return hiring_date


def generate_satisfaction(flexibility, overwork, salary, benefits, illness_days, age, expected_salary=None):
    """
    Calculates employee satisfaction based on various influencing factors.
    """
    base_satisfaction = 7
    flex_factor = (flexibility - 5) * 0.50
    overwork_factor = -0.5 * max(overwork - 5, 0) ** 1.2
    salary_factor = 0.2 * (salary / 1000)

    if expected_salary:
        salary_gap = expected_salary - salary
        if salary_gap > 0:
            salary_factor -= 0.05 * salary_gap / 1000

    benefits_factor = (benefits - 5) * 0.5
    illness_factor = -0.2 * illness_days ** 1.1
    age_factor = -0.3 if age > 60 else 0.2 if age < 30 else 0

    return max(1, min(10,
                      base_satisfaction + flex_factor + overwork_factor + salary_factor +
                      benefits_factor + illness_factor + age_factor))


def calculate_expected_salary(position, age, education_level, work_model, base_salary=30000):
    """
    Calculates the expected salary for an employee based on various parameters.
    """
    position_multiplier = {
        "Office Clerk": 1.0,
        "Specialist": 1.2,
        "Office Manager": 1.5,
        "Assistant Department Head": 1.8,
        "Department Head": 2.0,
        "Headquarters Manager": 2.5
    }
    pos_factor = position_multiplier.get(position, 1.0)

    age_factor = 1 + (max(age - 25, 0) // 5) * 0.05

    education_multiplier = {
        "Vocational Training": 1.0,
        "Bachelor Degree": 1.1,
        "Master Degree": 1.2,
        "Doctoral Candidate": 1.3
    }
    edu_factor = education_multiplier.get(education_level, 1.0)

    work_model_factor = 0.7 if work_model == "Part-time" else 0.9 if work_model == "Homeoffice" else 1.0

    expected_salary = base_salary * pos_factor * age_factor * edu_factor * work_model_factor

    return expected_salary
def adjust_for_inflation(base_salary, current_year, hire_year):
    """
    Adjusts salary for annual inflation (e.g., 2% per year).
    """
    inflation_rate = 0.02
    years_passed = max(current_year - hire_year, 0)
    return base_salary * ((1 + inflation_rate) ** years_passed)


def generate_illness_days(satisfaction, age, work_model, leadership_stress=False,
                          long_term_illness_prob=0.1, hire_date=None, current_date=None):
    """
    Generates the number of illness days, considering a limit in the first three months.

    :param satisfaction: Employee's satisfaction level.
    :param age: Employee's age.
    :param work_model: Work model (e.g., full-time or part-time).
    :param leadership_stress: Whether the position includes leadership responsibilities.
    :param long_term_illness_prob: Probability of long-term illnesses.
    :param hire_date: Employee's hire date.
    :param current_date: Current date.
    :return: Number of illness days.
    """
    # If hire_date and current_date are set, check if within the first three months
    if hire_date and current_date:
        hire_duration = (current_date - hire_date).days
        if hire_duration <= 90:  # First three months (90 days)
            illness_days = max(0, int(np.random.normal(loc=1, scale=1)))  # Small variance within the limit
            return min(3, illness_days)  # Maximum illness days in the first 3 months: 3

    # Normal calculation of illness days outside the first three months
    if random.random() < long_term_illness_prob:  # Long-term illnesses
        illness_days = max(
            15,
            int(np.clip(np.random.normal(loc=25 - satisfaction, scale=8), 15, 50))  # Increased scale (max. 50)
        )
    else:
        # Strengthened factors
        base_sickness = (5 + age // 8) / 12  # Age-weighted sickness base
        satisfaction_factor = ((10 - satisfaction) ** 2) * 0.5  # Low satisfaction more strongly negatively weighted
        flexibility_factor = -3 if work_model == "Part-time" else 0  # Part-time positively weighted
        age_factor = 4 if age > 55 else -2 if age < 25 else 0  # Strengthened age-based effects
        leadership_factor = 5 if leadership_stress else 0  # Stress from leadership responsibilities
        illness_days = int(np.clip(
            base_sickness + satisfaction_factor + flexibility_factor + age_factor + leadership_factor,
            0,  # No negative values
            40  # Upper limit
        ))

    return max(0, illness_days)


def decide_status(employee, satisfaction, overwork, performance_score,
                  salary, illness_days, tenure, training_costs, team_size, subordinates,
                  workplace_flexibility, job_role_progression, job_level,
                  switching_readiness):
    """
    Decides whether an employee remains active, retires, or terminates employment.

    :param employee: Employee's details dictionary.
    :param satisfaction: Employee's satisfaction level.
    :param overwork: Amount of overtime worked by the employee.
    :param performance_score: Employee's performance score.
    :param salary: Employee's salary.
    :param illness_days: Number of illness days.
    :param tenure: Employee's tenure (length of employment).
    :param training_costs: Training investments into the employee.
    :param team_size: The size of the team the employee works with.
    :param subordinates: Number of subordinates the employee has (if any).
    :param workplace_flexibility: Employee's workplace flexibility context.
    :param job_role_progression: Career development score of the employee.
    :param job_level: Employee's job level.
    :param switching_readiness: Employee's readiness to switch jobs externally.
    :return: "Active", "Retired", or "Terminated".
    """
    age = employee["Age"]

    termination_probability = 0.1  # Base probability for termination
    retirement_probability = 0.0  # Base probability for retirement

    # Age-related probabilities for retirement
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

    # Satisfaction
    if satisfaction <= 2:
        termination_probability += 0.45
    elif 2 < satisfaction <= 5:
        termination_probability += 0.2
    elif satisfaction >= 6:
        termination_probability -= (satisfaction - 5) * 0.08

    # Overtime
    if overwork > 15:
        termination_probability += 0.25 + 0.01 * (overwork - 15)
    elif overwork < 5:
        termination_probability -= 0.1

    # Salary
    expected_salary = 40000 + (age - 25) * 600
    if salary < expected_salary * 0.7:
        termination_probability += 0.3
    elif salary >= expected_salary * 1.3:
        termination_probability -= 0.1

    # Illness days
    if illness_days > 20:
        termination_probability += 0.15
    elif illness_days < 5:
        termination_probability -= 0.05

    # Training investments
    if training_costs > 2000:
        termination_probability -= 0.05

    # Team size and number of subordinates
    if team_size > 15 and employee["Position"] in ["Department Head", "Assistant Department Head", "Office Manager"]:
        termination_probability += 0.1
    if subordinates > 5:
        termination_probability += 0.05

    # Career development
    if job_role_progression <= 3:
        termination_probability += 0.2
    elif job_role_progression >= 7:
        termination_probability -= 0.1

    # Readiness to switch jobs
    if switching_readiness > 0.7:
        termination_probability += 0.2
    elif 0.4 <= switching_readiness <= 0.7:
        termination_probability += 0.1
    else:
        termination_probability -= 0.1

    # Combine probabilities
    combined_probability = retirement_probability + termination_probability
    random_roll = random.uniform(0, 1)

    # Decide status
    if random_roll < retirement_probability:
        return "Retired"
    elif random_roll < combined_probability:
        return "Terminated"
    else:
        return "Active"

def calculate_switching_readiness_with_emotions(satisfaction, overwork, salary, expected_salary, workplace_flexibility,
                                                illness_days, emotional_factors):
    """
    Calculates a weaker switching readiness based on objective factors and an emotional profile.
    - satisfaction: Satisfaction (1 to 10)
    - overwork: Number of overtime hours
    - salary: Actual salary
    - expected_salary: Expected salary
    - workplace_flexibility: Workplace flexibility (1 to 10)
    - illness_days: Number of illness days
    - emotional_factors: Emotional values (e.g., stress, conflicts, recognition, etc.)

    Returns:
        readiness: Switching readiness, a value between 0 and 1.
    """
    # Base value for switching readiness
    readiness = 0.3  # Slightly reduced from the previous value of 0.5

    # Incorporate satisfaction and overtime
    if satisfaction <= 4:
        readiness += (5 - satisfaction) * 0.08  # Slightly reduced weight
    if overwork > 10:
        readiness += (overwork - 10) * 0.005  # Overtime increases switching readiness, but more weakly

    # Salary
    if salary < expected_salary * 0.8:
        readiness += 0.15  # Reduced weight from 0.2
    elif salary >= expected_salary * 1.2:
        readiness -= 0.12  # Positive salary incentives weighted more

    # Workplace flexibility
    if workplace_flexibility >= 7:
        readiness -= (workplace_flexibility - 6) * 0.06  # Positive impact strengthened
    elif workplace_flexibility <= 3:
        readiness += (4 - workplace_flexibility) * 0.03  # Negative impact reduced

    # Illness days
    if illness_days > 10:
        readiness += (illness_days - 10) * 0.005  # Weight halved

    # Add emotional factors
    stress = emotional_factors.get("stress", 5)
    recognition = emotional_factors.get("recognition", 5)
    work_environment = emotional_factors.get("work_environment", 5)
    future_opportunities = emotional_factors.get("future_opportunities", 5)
    team_conflicts = emotional_factors.get("team_conflicts", 5)
    boredom = emotional_factors.get("boredom", 5)

    # Stress increases switching readiness (slightly weakened)
    readiness += (stress - 5) * 0.03

    # Recognition reduces switching readiness (slightly strengthened)
    readiness -= (recognition - 5) * 0.05

    # Positive work environment reduces switching readiness (slightly strengthened)
    readiness -= (work_environment - 5) * 0.04

    # Future opportunities weighted more strongly
    readiness -= (future_opportunities - 5) * 0.05

    # Conflicts increase switching readiness (slightly weakened)
    readiness += (team_conflicts - 5) * 0.04

    # Boredom (boreout) also increases switching readiness (slightly weakened)
    readiness += (boredom - 5) * 0.02

    # Ensure limits
    readiness = max(0, min(1, readiness))

    return round(readiness, 2)


# Step 1: Create fixed employee information
employee_ids = generate_employee_ids(num_employees)  # Generate employee IDs
persistent_employee_data = []  # Temporary list for all employee data

for i in range(num_employees):
    position = all_positions[i]  # Select position from the random list

    # Generate employee data (age and start date)
    employee_data = generate_employee_data(position, start_year, start_month)

    age = employee_data["Age"]
    birth_date = employee_data["Birthdate"]
    start_date = employee_data["Hiring Date"]  # Ensure the hiring date is set correctly

    # Consistent one-time calculations for education and work model
    education_level = random.choice(education_levels)  # Random selection of education level
    work_model = random.choice(work_models)  # Choose work model (e.g., full-time, part-time)

    # Calculate expected salary (consistent)
    expected_salary = calculate_expected_salary(
        position=position,
        age=age,
        education_level=education_level,
        work_model=work_model
    )

    # Create employee dataset
    employee = {
        "Employee_ID": employee_ids[i],
        "Name": fake.name(),
        "Gender": random.choice(genders),
        "Hiring Date": start_date,  # Either provided or generated date
        "Birthdate": birth_date,
        "Position": position,
        "Education Level": education_level,
        "Work Model": work_model,
        "Status": "Active",  # Initially always Active
        "Illness Days": generate_illness_days(
            satisfaction=random.randint(1, 10),  # Random satisfaction value
            age=age,
            work_model=work_model,
            leadership_stress=(position in leadership_positions)  # Leadership position?
        ),
        "Satisfaction": generate_satisfaction(
            flexibility=random.randint(0, 10),
            overwork=max(0, random.randint(0, 20)),
            salary=expected_salary,  # Use consistent salary
            benefits=random.randint(0, 10),
            illness_days=0,  # Initially without illness days
            age=age,
            expected_salary=expected_salary
        ),
        "Age": age,
        "Rehire Blocked": False,  # Always False initially
    }

    # Add employee data
    persistent_employee_data.append(employee)

# Pass the generated employee data
print(f"{num_employees} employees successfully initialized!")

# Step 2: Initialize succession planning
succession_planning = []

# Step 2: Begin the initial phase of time analysis
# Simulation starts with established employees
for year in years:
    for month in months:
        active_employees = [
            e for e in persistent_employee_data if e["Status"] == "Active"
        ]

        print(f"INFO: Year {year}, Month {month}, Active Employees: {len(active_employees)}")

        # Further preparatory calculations or logic can be added here,
        # before processing new hires or planned successors.

# Step 3: Generate monthly varying data
data = []
for year in years:
    for month in months:
        new_hires = []

        for entry in list(succession_planning):  # Make a copy of the succession plan to allow modifications
            if entry["Year"] == year and entry["Month"] == month:
                new_employee_id = random.randint(10000, 99999)
                while new_employee_id in employee_ids:
                    new_employee_id = random.randint(10000, 99999)
                employee_ids.append(new_employee_id)

                # Position from the succession plan
                position = entry["Position"]

                # Generate employee data with the current year and month
                employee_data = generate_employee_data(position, year, month)

                # Use age and birthdate directly from the generated data
                age = employee_data["Age"]
                birth_date = employee_data["Birthdate"]

                # NEW: Calculate expected salary
                expected_salary = calculate_expected_salary(
                    position=position,
                    age=age,
                    education_level=random.choice(education_levels),  # Randomly select education level
                    work_model=random.choice(work_models)  # Random work model
                )

                # Calculate illness days
                illness_days_calculated = generate_illness_days(
                    satisfaction=random.randint(1, 10),  # Random satisfaction rating
                    age=age,
                    work_model=random.choice(work_models),  # Random work model
                    leadership_stress=(position in leadership_positions),  # Check for leadership responsibility
                    hire_date=employee_data["Hiring Date"],  # Hire date of the new employee
                    current_date=date.today()  # Current date
                )

                # New employee dataset
                new_employee = {
                    "Employee_ID": new_employee_id,
                    "Name": fake.name(),
                    "Gender": random.choice(["male", "female", "diverse"]),
                    "Hiring Date": employee_data["Hiring Date"],
                    "Birthdate": birth_date,
                    "Age": age,
                    "Position": position,
                    "Status": "Active",

                    # Store illness days
                    "Illness Days": illness_days_calculated,
                    "Satisfaction": generate_satisfaction(
                        flexibility=random.randint(0, 10),
                        overwork=max(0, random.randint(0, 20)),  # Temporary value
                        salary=random.randint(25000, 80000),  # Actual salary
                        benefits=random.randint(0, 10),
                        illness_days=illness_days_calculated,  # Fix: Refer directly to the calculated illness days
                        age=age,
                        expected_salary=expected_salary  # Include the expected salary
                    ),
                    "Rehire Blocked": False,
                }

                # Create new hires and update succession plan
                new_hires.append(new_employee)
                succession_planning.remove(entry)

        # Remove old planned entries
        succession_planning = [
            entry for entry in succession_planning if
            entry["Year"] > year or (entry["Year"] == year and entry["Month"] > month)
        ]

        # Also consider planned successors from the succession plan
        planned_employees = sum(
            1 for entry in succession_planning if entry["Year"] == year and entry["Month"] >= month
        )
        active_employees = sum(1 for employee in persistent_employee_data if employee["Status"] == "Active")
        total_employees = active_employees + planned_employees

        # Check deviations in the number of employees
        deviation = num_employees - active_employees
        if abs(deviation) > 33:
            employees_to_add = random.randint(8, 19)

            for _ in range(employees_to_add):
                new_employee_id = random.randint(10000, 99999)
                while new_employee_id in employee_ids:
                    new_employee_id = random.randint(10000, 99999)
                employee_ids.append(new_employee_id)

                # Generate position and data
                position = random.choice(positions)
                employee_data = generate_employee_data(position, year, month)  # Correct arguments

                age = employee_data["Age"]
                birth_date = employee_data["Birthdate"]

                # NEW: Calculate expected salary
                expected_salary = calculate_expected_salary(
                    position=position,
                    age=age,
                    education_level=random.choice(education_levels),  # Randomly select education level
                    work_model=random.choice(work_models)  # Random work model
                )

                # Calculate illness days
                illness_days_calculated = generate_illness_days(
                    satisfaction=random.randint(1, 10),  # Random satisfaction rating
                    age=age,
                    work_model=random.choice(work_models),  # Random work model
                    leadership_stress=(position in leadership_positions),  # Check leadership responsibility
                    hire_date=employee_data["Hiring Date"],  # Hiring date of the new employee
                    current_date=date.today()  # Current date
                )

                # New employee dataset
                new_employee = {
                    "Employee_ID": new_employee_id,
                    "Name": fake.name(),
                    "Gender": random.choice(genders),
                    "Age": age,
                    "Hiring Date": employee_data["Hiring Date"],
                    "Birthdate": birth_date,
                    "Position": position,
                    "Education Level": random.choice(education_levels),
                    "Work Model": random.choice(work_models),
                    "Status": "Active",

                    # Save calculated illness days
                    "Illness Days": illness_days_calculated,

                    # Calculate satisfaction based on realistic values
                    "Satisfaction": generate_satisfaction(
                        flexibility=random.randint(0, 10),
                        overwork=max(0, random.randint(0, 20)),  # Temporary value
                        salary=random.randint(25000, 80000),  # Actual salary
                        benefits=random.randint(0, 10),
                        illness_days=illness_days_calculated,  # Reference calculated illness days
                        age=age,
                        expected_salary=expected_salary  # Add expected salary
                    ),

                    "Rehire Blocked": False,

                    # New fields for advanced logic
                    "Workplace Flexibility": random.randint(0, 10),  # Flexibility (0–10)
                    "Job Role Progression": random.randint(0, 10),  # Career development (0–10)
                    "Job Level": random.choice(list(range(1, 5))),  # Job level (1–4)
                    "Training Costs": random.randint(0, 5000) if random.random() < 0.5 else 0,  # Optional
                    "Team Size": random.randint(5, 25),  # Team size (based on position)
                    "Subordinates": max(0, positions.index(position) - 2)
                    if position in leadership_positions else 0  # Based on position
                }

                # Persist employee data
                persistent_employee_data.append(new_employee)

        # Output summary for the month
        print(
            f"INFO: {year}-{month:02d} completed -> Active: {active_employees}, "
            f"Planned: {planned_employees}, Total: {total_employees}, "
            f"New Hires: {len(new_hires)}"
        )

        for employee in persistent_employee_data:
            # Skip employees who are retired or have left
            if employee["Status"] in ["Retired", "Left"]:
                continue  # Skip the employee if not active

            # Calculate current date and age
            current_date = datetime.datetime(year, month, 1).date()
            employee["Age"] = (current_date.year - employee["Birthdate"].year) - (
                    (current_date.month, current_date.day) < (
                employee["Birthdate"].month, employee["Birthdate"].day)
            )

            # **Check: No data before the hiring date**
            if current_date < employee["Hiring Date"]:
                continue  # Skip this employee for dates before the hiring date

            # Calculate relevant values
            satisfaction = employee["Satisfaction"]  # Satisfaction value
            overwork = employee.get("Overtime", random.randint(0, 20))  # Dynamic overtime
            salary = employee.get("Salary", random.randint(25000, 80000))  # Salary
            expected_salary = calculate_expected_salary(
                position=employee["Position"],
                age=employee["Age"],
                education_level=employee["Education Level"],
                work_model=employee["Work Model"]
            )
            workplace_flexibility = {
                "Full-time": random.randint(3, 6),
                "Part-time": random.randint(6, 9),
                "Home Office": random.randint(7, 10)
            }.get(employee["Work Model"], random.randint(1, 5))  # Default if not defined

            # Update illness days
            leadership_stress = employee["Position"] in leadership_positions
            employee["Illness Days"] = generate_illness_days(
                satisfaction=satisfaction,
                age=employee["Age"],
                work_model=employee["Work Model"],
                leadership_stress=leadership_stress
            )

            # Initialize emotional factors (if not present)
            emotional_factors = employee.get("Emotional Factors", {
                "stress": random.randint(1, 10),
                "recognition": random.randint(1, 10),
                "work_environment": random.randint(1, 10),
                "future_opportunities": random.randint(1, 10),
                "team_conflicts": random.randint(1, 10),
                "boredom": random.randint(1, 10),
            })

            # Calculate switching readiness
            switching_readiness = calculate_switching_readiness_with_emotions(
                satisfaction=satisfaction,
                overwork=overwork,
                salary=salary,
                expected_salary=expected_salary,
                workplace_flexibility=workplace_flexibility,
                illness_days=employee["Illness Days"],
                emotional_factors=emotional_factors
            )

            # Set standard status
            new_status = "Active"  # Default to "Active"

            # Retirement at 67 years
            if employee["Age"] >= 67:
                employee["Status"] = "Retired"
                employee["Exit Date"] = current_date
                continue

            # Retirement probability (between 60 and 69 years)
            elif 60 <= employee["Age"] < 70:
                retirement_probability = max(0.05, 0.05 * (5 - satisfaction))  # Base 5% probability
                if random.random() < retirement_probability:
                    new_status = "Retired"

            # Resignation logic based on switching readiness and low satisfaction
            elif switching_readiness > 0.7 and satisfaction < 4:
                new_status = "Left"

            # Resignation probability for long-term dissatisfied employees
            elif employee["Hiring Date"].year < year - 5 and satisfaction < 5 and random.random() < 0.1:
                new_status = "Left"

            # Update the employee's status
            employee["Status"] = new_status

            # Ensure retirement rules after 67 years are correctly enforced
            if employee["Age"] >= 67 and employee["Status"] != "Retired":
                print(f"DEBUG: Faulty status update for employee {employee['Employee_ID']}.")
                employee["Status"] = "Retired"

            # Exit date and succession planning for "Left" or "Retired"
            if employee["Status"] in ["Left", "Retired"]:
                employee["Exit Date"] = current_date
                employee["Exit Year"] = year
                employee["Exit Month"] = month
                employee["Last Active Month"] = f"{year:04d}-{month:02d}"

                # NEW BLOCK: Direct succession planning for this employee
                succession_planning.append({
                    "Year": year,
                    "Month": month + 1 if month < 12 else 1,
                    "Position": employee["Position"]
                })

                # Plan successors, if necessary
                if deviation > 0:
                    hiring_month = month + 1
                    hiring_year = year
                    if hiring_month > 12:
                        hiring_month -= 12
                        hiring_year += 1
                    for _ in range(min(deviation, 15)):  # Plan up to 15 successors
                        position = random.choice(positions)
                        gender = random.choice(genders)
                        succession_planning.append({
                            "Year": hiring_year,
                            "Month": hiring_month,
                            "Position": position,
                            "Gender": gender,
                        })
                        print(
                            f"DEBUG: Successor planned. Position: {position}, Hiring: {hiring_year}-{hiring_month:02d}")

            # Enhanced correlations for illness days
            current_date = datetime.datetime(year, month, 1).date()
            birth_date = employee["Birthdate"]
            age = (current_date.year - birth_date.year) - (
                    (current_date.month, current_date.day) < (birth_date.month, birth_date.day)
            )

            leadership_stress = employee["Position"] in leadership_positions

            # Calculate illness days using the function
            illness_days = generate_illness_days(
                satisfaction=employee.get("Satisfaction", 7),
                age=age,
                work_model=employee.get("Work Model", "Full-time"),
                leadership_stress=leadership_stress
            )

            # Dynamically choose absence reason
            if illness_days > 0:
                if illness_days > 20:  # Longer illness days
                    absence_reason = "Illness with rehab"
                else:
                    absence_reason = random.choices(
                        absence_reasons,
                        weights=[50 - age // 10, 30 + (age // 20), 10, 10],  # Weighting based on age
                        k=1
                    )[0]
            else:
                absence_reason = "none"

            # Calculate salary based on an irregular distribution and satisfaction
            base_salary = (
                random.randint(30000, 45000) if employee["Position"] in ["Clerk", "Specialist"]
                else random.randint(50000, 80000)
            )

            # Salary adjustment: Irregular and random
            irregularity_factor = random.triangular(0.8, 1.2, 1.5)  # Asymmetric distribution
            inflation_adjustment = base_salary * ((year - 2015) * 0.02)  # 2% inflation per year
            satisfaction_bonus = employee.get("Satisfaction", 5) * 500  # Satisfaction influences salary

            salary = (base_salary * irregularity_factor) + inflation_adjustment + satisfaction_bonus

            base_data = {
                "Year": year,
                "Month": month,
                "Employee_ID": employee["Employee_ID"],
                "Name": employee["Name"],
                "Gender": employee["Gender"],
                "Hiring Date": employee["Hiring Date"],
                "Exit Date": employee.get("Exit Date", None),
                "Position": employee["Position"],
                "Education Level": employee["Education Level"],
                "Age": age,
                "Birthdate": employee["Birthdate"],
                "Salary Group": salary_groups[employee["Position"]],
                "Basic Education": random.choice(["Yes", "No"]),
                "Internal Training": random.randint(0, 5),
                "Planned Position": random.choice(["Yes", "No"]),
                "Time Until Retirement": max(0, 67 - age),
                "Married": random.choice(["Yes", "No"]),
                "Children": random.choice(["Yes", "No"]),
                "Vacation Days Taken": random.randint(0, 30 // 12),
                "Overtime": max(0,
                                int(np.random.normal(
                                    loc=(15 + positions.index(employee["Position"]) * 5),
                                    scale=5  # Larger spread for more realistic distribution
                                ))),
                "Absence Days": illness_days,
                "Absence Reason": absence_reason,
                "Annual Performance Review": random.randint(1, 5),
                "Location": random.choices(locations, weights=[70, 20, 10])[0],
                "Switching Readiness": employee.get("Switching Readiness", round(random.uniform(0, 1), 2)),
                "Training Costs": employee.get("Training Costs",
                                               random.randint(0, 5000) if random.random() < 0.5 else 0),
                "Number of Subordinates": employee.get("Number of Subordinates",
                                                       max(0, positions.index(employee["Position"]) - 2)
                                                       if employee["Position"] in leadership_positions else 0),
                "Job Role Progression": employee.get("Job Role Progression", random.choices(
                    ["Promotion", "Lateral Move", "Leave and Rejoin", "No Change"],
                    weights=[30, 20, 10, 40])[0]),
                "Salary": salary,
                "Workplace Flexibility": employee.get("Workplace Flexibility", random.randint(0, 10)),
                "Team Size": employee.get("Team Size", random.randint(5, 25)),
                "Coaching": random.choice(["Yes", "No"]),
                "Job Level": employee.get("Job Level", random.choice(job_levels)),
                "Work Model": employee["Work Model"],
                "Satisfaction": employee["Satisfaction"],
                "Status": employee["Status"],
                "Tenure": year - employee["Hiring Date"].year
            }

            # Keep children, basic education, and status consistent
            if year > min(years) or month > min(months):  # Check for all times except the first month/year
                # Check previous datasets
                prev_entry = next(
                    (entry for entry in data if
                     entry.get("Year") == year and
                     entry.get("Month") == month - 1 and
                     entry.get("Employee_ID") == employee.get("Employee_ID")),
                    None
                )

                if not prev_entry:  # If the previous month doesn't exist, check for the previous year's December
                    prev_entry = next(
                        (entry for entry in data if
                         entry.get("Year") == year and
                         entry.get("Month") == month - 1 and
                         entry.get("Employee_ID") == employee.get("Employee_ID")),
                        None
                    )
                if prev_entry:
                    if prev_entry["Children"] == "Yes":
                        base_data["Children"] = "Yes"
                    if prev_entry["Basic Education"] == "Yes":
                        base_data["Basic Education"] = "Yes"

            data.append(base_data)

# Step 3: Create DataFrame
df_updated = pd.DataFrame(data)

# Validate all hiring dates and ensure only the date (without time) is stored
df_updated['Hiring Date'] = pd.to_datetime(df_updated['Hiring Date'], errors='coerce', format='%Y-%m-%d')
df_updated['Exit Date'] = pd.to_datetime(df_updated['Exit Date'], errors='coerce', format='%Y-%m-%d')
df_updated['Birthdate'] = pd.to_datetime(df_updated['Birthdate'], errors='coerce', format='%Y-%m-%d')

# Save data without time information
df_updated.to_csv("../data/HR_dataset.csv", index=False, date_format="%Y-%m-%d")

end_time = time.time()
print(f"Analysis completed in {end_time - start_time:.2f} seconds.")


