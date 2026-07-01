from tools.validator import ClaimValidator

validator = ClaimValidator()

sample_claim = {

    "employee_id": "EMP001",

    "expense_type": "Hotel",

    "amount": 7500,

    "currency": "INR",

    "expense_date": "2026-06-20",

    "description": "Hotel stay during client meeting in Bangalore.",

    "receipt_uploaded": True,

}

result = validator.validate(sample_claim)

print("=" * 60)

print("VALIDATION RESULT")

print("=" * 60)

for key, value in result.items():

    print(f"{key} : {value}")