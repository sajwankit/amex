def seeder(torch=None, np=None, random=None, py=True, seed=42):
    if random:
        random.seed(seed)
    if py:
        os.environ['PYTHONHASHSEED'] = str(seed)
    if np:
        np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    seed_summary = ''
    for x, str_x in {torch:'torch', np:'np', py: 'py', random:'random'}.items():
        if x:
            seed_summary = f'{seed_summary} {str_x} seeded with seed: {seed} | '
    return seed_summary