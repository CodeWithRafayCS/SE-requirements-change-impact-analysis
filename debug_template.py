from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, yaml

cfg = yaml.safe_load(open("config_demo.yaml"))
tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-base").eval()

# A pair that SHOULD have high entailment under a *good* template
change = "Add a new discount type that only applies to first-time customers."
req_related = "The system shall apply a discount code to the order total before payment."
req_unrelated = "The system shall allow an administrator to deactivate a user account."

prem_t = cfg["nli"]["premise_template"]
hyp_t  = cfg["nli"]["hypothesis_template"]

for label, req in [("related", req_related), ("unrelated", req_unrelated)]:
    premise = prem_t.format(change_text=change)
    hyp = hyp_t.format(requirement_text=req)
    enc = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    print(f"--- {label} ---")
    print(f"  entailment={probs[1]:.4f}  neutral={probs[2]:.4f}  contra={probs[0]:.4f}")