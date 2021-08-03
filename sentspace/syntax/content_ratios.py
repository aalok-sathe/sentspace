

def get_content_ratio(sent_num, tag_list):
    """
    Given sentence number and boolean tag corresponding to this sentence's word, calculate the content ratio
    """
    df = pd.DataFrame({'sent_num': sent_num, 'tag_list': tag_list})

    # Per sentence get the ratio
    cont_ratios = {}
    for sent in df.sent_num.unique():
        # Grab just the specific sentences
        t_df = df[df.sent_num == sent]['tag_list']

        # Get the ratio of content to total
        cont_ratios[sent] = t_df.sum()/len(t_df)

    df_cont_ratio = pd.DataFrame(
        {'sent_num': cont_ratios.keys(), 'content_ratio': cont_ratios.values()})
    return df_cont_ratio
