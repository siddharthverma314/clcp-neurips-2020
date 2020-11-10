;; -*- lexical-binding: t -*-
(require 's)
(require 'f)

;; Launch variables

(defcustom launch--google-credential-file
  "/home/vsiddharth/secrets/vsiddharth.json"
  "Location of GCE auth.json file for gcloud")

(defcustom launch--google-project
  "kelvinxu-research"
  "Name of current GCE project")

(defcustom launch--machine-type
  "n1-standard-4"
  "Type of GCE instance to create")

(defcustom launch--zones
'("us-west1-a" "us-west1-b")
  "Zones to launch GCE in")

(defcustom launch--disk-size
  50
  "Disk size of GCE")

(defcustom launch-docker-image
  "siddharthverma/adversarial"
  "Image that is run on GCP")


;; thin wrapper around docker-machine
(defun launch--setup-env ()
  "Setup docker-machine environment variables to use google as a driver"

  (setenv "MACHINE_DRIVER" "google")
  (setenv "GOOGLE_APPLICATION_CREDENTIALS" launch--google-credential-file)
  (setenv "GOOGLE_PROJECT" launch--google-project)
  (setenv "GOOGLE_MACHINE_TYPE" launch--machine-type)
  (setenv "GOOGLE_ZONE" (nth (random (length launch--zones)) launch--zones))
  (setenv "GOOGLE_DISK_SIZE" (number-to-string launch--disk-size)))


(defmacro launch--cmd (name &rest command)
  (launch--setup-env)
  `(start-process ,name
		  (format "*docker-machine-%s*" ,name)
		  "docker-machine"
		  ,@command))

(defun launch-cmd-create (instance-name)
  (sleep-for (/ (random 100) 10)) ;; required so that google does not rate limit
  (launch--cmd instance-name "create" instance-name))

(defun launch-cmd-ssh (instance-name command)
  (message (format "LAUNCH SSH: %s" command))
  (launch--cmd instance-name "ssh" instance-name command))

(defun launch-cmd-scp-to (instance-name file remote-location &optional recursive)
  (let ((location (format "%s:%s" instance-name remote-location)))
    (if recursive
	(launch--cmd instance-name "scp" "-r" file location)
      (launch--cmd instance-name "scp" file location))))


;; launch helpers
(defmacro launch-chain-commands (&rest commands)
  "Chain a bunch of commands that each return processes"
  (cl-labels
      ((chain (commands)
	      (if (null commands) nil
		`(set-process-sentinel
		  ,(car commands)
		  (lambda (_ _)
		    ,(chain (cdr commands)))))))
    (chain commands)))

(defun launch--make-args (args)
  "Format arguments into command line"
  (s-join
   " "
   (mapcar
    (lambda (arg)
      (cond
       ((= (length arg) 1) (format "--%s" (symbol-name (car arg))))
       ((= (length arg) 2) (format "--%s=%s" (symbol-name (car arg)) (cadr arg)))
       (t (error (format "Invalid argument %s" arg)))))
    args)))

(cl-defun launch--make-docker-cmd
    (script-name args &optional (dir-mappings '(("/log" "/log"))))
  (format "sudo docker run %s -d %s %s %s"
	  (s-join " " (--map (s-prepend "-v " (s-join ":" it)) dir-mappings))
	  launch-docker-image
	  script-name
	  (launch--make-args (append args '((logdir "/log") (device "cpu"))))))

(defun launch--compose-cmds (&rest commands)
  (s-join "; " commands))


;; actual launch functions
(defun launch-run-regular (instance-name script-name args)
  (let* ((docker-command (launch--make-docker-cmd script-name args))
	 (launch-command (launch--compose-cmds "sudo mkdir /log" docker-command)))
    (launch-chain-commands
     (launch-cmd-create instance-name)
     (launch-cmd-ssh instance-name launch-command))))


(defun launch-run-hrl (instance-name script-name file &optional args)
  (let* ((docker-command
	  (launch--make-docker-cmd
	   script-name
	   (append args `((checkpoint ,(f-join "/checkpoint" (f-filename file)))))
	   '(("/log" "/log") ("/home/docker-user/checkpoint" "/checkpoint"))))
	 (launch-command (launch--compose-cmds "sudo mkdir /log" docker-command)))
    (launch-chain-commands
     (launch-cmd-create instance-name)
     (launch-cmd-scp-to instance-name (f-parent file) "~/checkpoint" t)
     (launch-cmd-ssh instance-name launch-command))))


(provide 'launch)
