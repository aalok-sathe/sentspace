

def sanity_check_databases(databases):
    '''
    perform sanity checks upon loading various datasets
    to ensure validity of the loaded data
    '''

    assert databases['NRC_Arousal']['happy'] == 0.735
    assert databases['NRC_Valence']['happy'] == 1
    assert databases['OSC']['happy'] == 0.951549893181384
    assert abs(databases['aoa']['a'] - 2.893384) < 1e-4
    assert databases['concreteness']['roadsweeper'] == 4.85
    # assert abs(databases['imag']['abbey'] - 5.344) < 1e-4
    assert databases['total_degree_centrality']['a'] == 30
    assert databases['lexical_decision_RT']['a'] == 798.917
    assert abs(databases['log_contextual_diversity']['a'] - 3.9234) < 1e-4
    assert abs(databases['log_lexical_frequency']['a'] - 6.0175) < 1e-4
    assert databases['n_orthographic_neighbors']['a'] == 950.59
    assert databases['num_morpheme']['abbreviated'] == 4
    assert abs(databases['prevalence']['a'] - 1.917) < 1e-3
    assert databases['surprisal-3']['beekeeping'] == 10.258
