from ucimlrepo import fetch_ucirepo
orig_df = fetch_ucirepo(id=848)['data']['original']
orig_df.to_csv("orig.csv")