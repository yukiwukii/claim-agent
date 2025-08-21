from src.nodes.binary_classifier import binary_classifier
with open("src/tests/txts/claim_form.txt", "r") as file:
    document = file.read()

state = {"current_document": document}
result = binary_classifier(state)

print(result.claim_form_or_not)