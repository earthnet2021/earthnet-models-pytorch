'use strict';

const CopyPlugin = require('copy-webpack-plugin');
const { parse } = require('./lib/parse.cjs');
const { validate } = require('./lib/schema.cjs');

class CopyWebpack {
  constructor(options = []) {
    validate(options);
    this.options = options;
  }

  apply(compiler) {
    const plugin = new CopyPlugin({
      patterns: parse(this.options),
    });

    plugin.apply(compiler);
  }
}

module.exports = { CopyWebpack };
