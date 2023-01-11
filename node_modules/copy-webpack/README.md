# copy-webpack

A simple [copy-webpack-plugin].

[![npm][npm-badge]][npm-url]
[![github][github-badge]][github-url]
![node][node-badge]

[copy-webpack-plugin]: https://github.com/webpack-contrib/copy-webpack-plugin
[npm-url]: https://www.npmjs.com/package/copy-webpack
[npm-badge]: https://img.shields.io/npm/v/copy-webpack.svg?style=flat-square&logo=npm
[github-url]: https://github.com/best-shot/copy-webpack
[github-badge]: https://img.shields.io/npm/l/copy-webpack.svg?style=flat-square&colorB=blue&logo=github
[node-badge]: https://img.shields.io/node/v/copy-webpack.svg?style=flat-square&colorB=green&logo=node.js

## Installation

```bash
npm install copy-webpack --save-dev
```

## Usage

```cjs
// example: webpack.config.cjs
const { CopyWebpack } = require('copy-webpack');

module.exports = {
  plugins: [new CopyWebpack('static')]
};
```

## Options

Glob or path from where we copy files.

`string` `object` `[string|object]`

```diff
- new CopyWebpackPlugin({
-   patterns: [
-     {
-       from: 'static',
-       globOptions: {
-         dot: true,
-         ignore: ['.gitkeep']
-       }
-     }
-   ]
- });
+ new CopyWebpack('static');
```

```diff
- new CopyWebpackPlugin({
-   patterns: [
-     {
-       from: 'extra',
-       noErrorOnMissing: true,
-       globOptions: {
-         dot: true,
-         ignore: ['.gitkeep']
-       }
-     }
-   ]
- });
+ new CopyWebpack([
+   {
+     from: 'extra',
+     noErrorOnMissing: true
+   }
+ ]);
```
