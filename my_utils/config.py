p = {}

p['logs_dir'] = "./Logs/"
p['subspace_dir'] = "./subspace/subspace_cache/"
p['epochs'] = 10
p['save_dir'] = "./trained_models/"
p["proj_prob"] = 0.5

labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label = {id_: label for id_, label in enumerate(labels)}
label_2_id = {label: id_ for id_, label in enumerate(labels)}
