---
id: cl52xnwmu50hzflibt1ksqw
title: VsCode_Vim_keybindings_configs
desc: ''
updated: 1740419933378
created: 1740419769850
---

## keybindings.json

`ctrl` + `hjkl` 导航 autosuggestion：

```json
{
  "key": "ctrl+j",
  "command": "selectNextSuggestion",
  "when": "suggestWidgetVisible"
},
{
  "key": "ctrl+k",
  "command": "selectPrevSuggestion",
  "when": "suggestWidgetVisible"
},{
  "key": "ctrl+j",
  "command": "workbench.action.quickOpenSelectNext",
  "when": "inQuickOpen"
},
{
  "key": "ctrl+k",
  "command": "workbench.action.quickOpenSelectPrevious",
  "when": "inQuickOpen"
}
```

## Ref and Tag

#Vim