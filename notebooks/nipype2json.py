""" nipype2json.py

Makes a Porcupine-compatible dictionary of nodes.
Created by Lukas Snoek (University of Amsterdam) 2017, Tim van Mourik (Donders 2019)

"""
import inspect
import importlib
import os.path as op
from copy import copy


def node2json(node, node_name=None, module=None, custom_node=False,
              module_path=None):
    if node_name is None and custom_node:
        raise ValueError("Cannot infer node-name from custom-nodes! Please "
                         "set the argument `node_name` correctly!")

    if node_name is None:
        node_name = _get_node_name(node)

    all_inputs, mandatory_inputs = _get_inputs(node, custom_node)
    all_outputs = _get_outputs(node, custom_node)
    descr = _get_descr(node, node_name, custom_node)

    if custom_node:
        toolbox = 'Custom'
    else:
        toolbox = 'Nipype'

    this_category = []
    if module.split('.')[0] == 'algorithms':
        this_category.append('algorithms')

    if custom_node:
        this_category.append(module)
    else:
        this_category.append(module.split('.')[1])

    interface_name = copy(this_category)
    if not custom_node:
        sub_modules = _get_submodule(node)[1:]
        if sub_modules and sub_modules[0] != this_category[-1]:
            this_category.extend(sub_modules)

    web_url = _get_web_url(node, module, custom_node)
    import_statement = _get_import_statement(node, module, module_path)
    init_statement = _get_init_statement(interface_name, node_name, custom_node)
    code = [{
        'language': 'Nipype',
        'comment': descr,
        'argument': {
            "name": init_statement,
            "import": import_statement
        }
    }]
    code.append({
        'language': 'Docker',
        'argument': {
            "name": ", ".join(interface_name)
        }
    })
    ports = []
    for inp in all_inputs:
        codeBlock = {
            'language': 'Nipype',
            'argument': {
                "name": inp
            }
        }

        is_mandatory = inp in mandatory_inputs
        port = {
            'input': True,
            'output': False,
            'visible': True if is_mandatory else False,
            'editable': True,
            'name': inp,
            'code': [codeBlock]
        }

        ports.append(port)

    ports = sorted(ports, reverse=True, key=lambda p: p['visible'])
    for outp in all_outputs:
        codeBlock = {
            'language': 'Nipype',
            'argument': {
                "name": outp
            }
        }
        port = {
            'input': False,
            'output': True,
            'visible': True,
            'editable': False,
            'name': outp,
            'code': [codeBlock]
        }
        ports.append(port)

    node_to_return = {
        'toolbox': toolbox,
        'name': '%s.%s' % (interface_name[-1], node_name),
        'web_url': web_url,
        'category': this_category,
        'code': code,
        'ports': ports
    }
    return node_to_return


def _get_inputs(node, custom_node=True):
    all_inputs, mandatory_inputs = [], []
    if custom_node:
        TO_SKIP = ['function_str', 'trait_added', 'trait_modified',
                   'ignore_exception']
        all_inputs.extend([inp for inp in node.inputs.traits().keys()
                           if inp not in TO_SKIP])
        mandatory_inputs.extend(all_inputs)
    else:
        all_inputs.extend([inp for inp in node.input_spec().traits().keys()
                           if not inp.startswith('trait')])
        mandatory_inputs.extend(node.input_spec().traits(mandatory=True).keys())

    return all_inputs, mandatory_inputs


def _get_outputs(node, custom_node=True):
    if custom_node:
        TO_SKIP = ['trait_added', 'trait_modified']
        outputs = list(node.aggregate_outputs().traits().keys())
        all_outputs = [outp for outp in outputs
                       if not outp in TO_SKIP]
    else:
        if hasattr(node, 'output_spec'):
            if node.output_spec is not None:
                all_outputs = [outp for outp in node.output_spec().traits().keys()
                               if not outp.startswith('trait')]
            else:
                all_outputs = []
        else:
            all_outputs = []

    return all_outputs


def _get_descr(node, node_name, custom_node):
    if custom_node:
        descr = 'Custom interface wrapping function %s' % node_name
    else:
        if hasattr(node, 'help'):
            descr = node.help(returnhelp=True).splitlines()[0]
        else:
            descr = node.__name__

    return descr


def _get_web_url(node, module, custom_node):
    if custom_node:
        return ''

    is_algo = module.split('.')[0] == 'algorithms'
    web_url = 'https://nipype.readthedocs.io/en/latest/interfaces/generated/'
    all_sub_modules = _get_submodule(node)
    if is_algo or len(all_sub_modules) < 2:
        module = 'nipype.' + module

    web_url += module
    if len(all_sub_modules) > 1:
        if not is_algo:
            web_url += '/%s.html' % all_sub_modules[1]
        else:
            web_url += '.html'
        web_url += '#%s' % node.__name__.lower()
    else:
        web_url += '.html#%s' % node.__name__.lower()

    return web_url


def _get_node_name(node):

    return node.__name__


def _get_import_statement(node, module, module_path):

    try:
        importlib.import_module('nipype.' + module)
        import_statement = "import nipype.%s as %s" % (module, module.split('.')[-1])
    except ImportError:

        import_statement = ''
        if module_path is not None:
            import_statement += "sys.path.append('%s')\n" % (op.abspath(op.dirname(module_path)))

        import_statement += 'import %s' % module

    return import_statement


def _get_init_statement(interface_name, node_name, custom_node):

    if custom_node:
        init_statement = interface_name[-1] + '.%s' % node_name
    else:
        init_statement = interface_name[-1] + '.%s()' % node_name

    return init_statement


def _get_submodule(node):

    module_tree = inspect.getmodule(node).__name__
    all_sub_modules = [n for n in module_tree.split('.')
                       if n not in ('interfaces', 'nipype')]
    return all_sub_modules
