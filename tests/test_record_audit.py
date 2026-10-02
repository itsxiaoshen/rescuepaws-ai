"""Record audit tests with a fake LLM: no API calls."""
from src.data_loader import load_animals
from src.record_audit import ConcernCheck, audit_animal


def animal(animal_id):
    return next(a for a in load_animals() if a.animal_id == animal_id)


def fake_check(concern_found, quote):
    calls = []

    def generate(system, user, output_type):
        calls.append(user)
        return ConcernCheck(concern_found=concern_found, quote=quote, explanation="test")

    generate.calls = calls
    return generate


def test_only_fields_recorded_as_yes_are_checked():
    # Otis: good_with_children yes, good_with_cats yes, good_with_dogs unknown -> 2 checks
    fake = fake_check(False, "")
    audit_animal(animal("RP-0019"), generate=fake)
    assert len(fake.calls) == 2
    assert not any("OTHER DOGS" in call for call in fake.calls)


def test_concern_with_real_quote_is_reported():
    quote = "he growled when a visiting toddler reached toward his food bowl."
    conflicts = audit_animal(animal("RP-0019"), generate=fake_check(True, quote))
    assert conflicts and conflicts[0]["quote"] == quote


def test_concern_with_made_up_quote_is_dropped():
    conflicts = audit_animal(animal("RP-0019"), generate=fake_check(True, "He bit a child."))
    assert conflicts == []
