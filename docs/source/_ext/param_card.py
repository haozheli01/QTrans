from docutils import nodes
from docutils.parsers.rst import Directive

class ParamCard(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    option_spec = {
        'options': str,
        'default': str,
        'block': str,
    }

    def run(self):
        name = self.arguments[0]
        param_options = self.options.get('options', '')
        default = self.options.get('default', '')
        block = self.options.get('block', '')
        block_str = ' '.join(block.split('_'))

        # create container with classes including block name
        container = nodes.container(classes=['param-card', block])

        # param-header
        header_html = f'<div class="param-header">{name}</div>'
        container += nodes.raw('', header_html, format='html')

        # param-meta
        meta_html = (
            f'<div class="param-meta">'
            f'Options: <code>{param_options}</code><br>'
            f'Default: <code>{default}</code>'
            f'</div>'
        )
        container += nodes.raw('', meta_html, format='html')

        # param-desc: using nested_parse to support rst languages
        desc_container = nodes.container(classes=['param-desc'])
        self.state.nested_parse(self.content, self.content_offset, desc_container)
        container += desc_container

        # param-block
        block_html = f'<div class="param-block">{block_str}</div>'
        container += nodes.raw('', block_html, format='html')

        return [container]

def setup(app):
    app.add_directive("param-card", ParamCard)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
