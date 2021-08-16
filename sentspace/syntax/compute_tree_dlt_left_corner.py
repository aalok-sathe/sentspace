import sentspace.syntax

text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")

features = sentspace.syntax.get_features(text, dlt=True, left_corner=True)
print(features.dlt)
print(features.tree)
print(features.left_corner)

# output = utils.pipeline(text, dlt=True)  # === output = utils.pipeline(text, dlt=True, left_corner = False)
# print(output.left_corner)  # is None
