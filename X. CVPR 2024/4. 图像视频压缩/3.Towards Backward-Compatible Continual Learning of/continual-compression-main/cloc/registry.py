__all__ = ['register_model', 'get_model', 'register_dataset', 'get_dataset']

_root_registry = dict()
_root_registry['model'] = dict()
_root_registry['dataset'] = dict()


def _register(group, func):
    name = func.__name__
    sub_registry = _root_registry[group]
    if name in sub_registry:
        existing = sub_registry[name]
        msg = f'Warning: *{name}* is already defined in {group=}. {existing=}, {func=}.'
        print(f'\u001b[93m' + msg + '\u001b[0m')
    sub_registry[name] = func
    return func

def _get(group, name, *args, **kwargs):
    sub_registry = _root_registry[group]
    assert name in sub_registry, f'Unknown {group} name: {name}'
    func = sub_registry[name]
    return func(*args, **kwargs)


def register_model(func):
    return _register('model', func)

def get_model(name, *args, **kwargs):
    return _get('model', name, *args, **kwargs)


def register_dataset(func):
    return _register('dataset', func)

def get_dataset(name, *args, **kwargs):
    return _get('dataset', name, *args, **kwargs)
