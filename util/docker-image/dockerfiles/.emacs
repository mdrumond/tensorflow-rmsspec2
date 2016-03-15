

; Do not create *~ or #*#
(setq make-backup-files nil)
(setq auto-save-default nil) ; stop creating #autosave# files

					; Save open files
(desktop-save-mode 1)

(require 'package)
(package-initialize)

(add-to-list 'package-archives
	     '("melpa" . "http://melpa.milkbox.net/packages/") t)

      ; Jedi - auto complete
(add-hook 'python-mode-hook 'jedi:setup)
(setq jedi:complete-on-dot t)                 ; optional

					; Show line number and column number
(global-linum-mode 1)
(column-number-mode 1)
;; Highlight parenteisis
(show-paren-mode 1) ; turn on paren match highlighting
(setq show-paren-style 'expression) ; highlight entire bracket expression

(setq-default indent-tabs-mode nil) ; no tabs

(setq default-tab-width 4) ;

;; Emacs auto completion
(autoload 'bash-completion-dynamic-complete
  "bash-completion"
  "BASH completion hook")
(add-hook 'shell-dynamic-complete-functions
	  'bash-completion-dynamic-complete)
