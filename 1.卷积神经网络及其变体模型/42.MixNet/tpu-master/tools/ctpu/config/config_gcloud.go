// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

package config

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/user"
	"path"
	"strings"

	"github.com/zieckey/goini"
)

const appDefaultFile = "application_default_credentials.json"

func gcloudConfig() (*Config, error) {
	homedir := ""
	if configDir, ok := os.LookupEnv("CLOUDSDK_CONFIG"); ok {
		return buildGcloudEnvConfig(configDir, true)
	}
	user, err := user.Current()
	if err != nil {
		// Fallback to looking up in the environment as a final attempt.
		var ok bool
		homedir, ok = os.LookupEnv("HOME")
		if !ok {
			return nil, err
		}
	} else {
		homedir = user.HomeDir
	}
	if len(homedir) == 0 {
		return nil, errors.New("could not find your home directory")
	}

	// Set this flag to ignore checking whether ~/.config/gcloud/application_default_credentials.json exists.
	// Useful for environments where there is a service account/metadata server available, but the file does not exist.
	// Examples include special Docker containers.
	ignoreFlag, ignoreAppCredentialsFile := os.LookupEnv("CTPU_IGNORE_APP_CREDENTIALS_FILE")
	if ignoreAppCredentialsFile && (ignoreFlag == "0" || ignoreFlag == "false") {
		ignoreAppCredentialsFile = false
	}

	return buildGcloudEnvConfig(path.Join(homedir, ".config", "gcloud"), !ignoreAppCredentialsFile)
}

func buildGcloudEnvConfig(configDir string, checkAppCredentials bool) (*Config, error) {
	if stat, err := os.Stat(configDir); os.IsNotExist(err) || !stat.IsDir() {
		return nil, fmt.Errorf("expected gcloud config directory at '%s'", configDir)
	}

	if err := checkAppDefaultFile(configDir); err != nil && checkAppCredentials {
		return nil, err
	}

	if !hasGcloudConfig(configDir) {
		return &Config{}, nil
	}

	activeConfigBytes, err := ioutil.ReadFile(path.Join(configDir, "active_config"))
	if err != nil {
		return nil, fmt.Errorf("error reading gcloud configuration (no active config) %#v", err)
	}
	activeConfig := strings.TrimSpace(string(activeConfigBytes))

	activeConfigFile := path.Join(configDir, "configurations", "config_"+activeConfig)
	if _, err := os.Stat(activeConfigFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("error reading gcloud configuration (active config file not found) %#v", err)
	}

	ini := goini.New()
	err = ini.ParseFile(activeConfigFile)
	if err != nil {
		return nil, fmt.Errorf("unable to parse configuration file '%s': %#v", activeConfigFile, err)
	}

	account, _ := ini.SectionGet("core", "account")
	project, _ := ini.SectionGet("core", "project")
	zone, _ := ini.SectionGet("compute", "zone")

	return &Config{
		Environment:         "gcloud",
		account:             account,
		ActiveConfiguration: activeConfig,
		Project:             project,
		Zone:                zone,
	}, nil
}

func checkAppDefaultFile(configDir string) error {
	// TODO(saeta): bypass this check if running within GCE
	stat, err := os.Stat(path.Join(configDir, appDefaultFile))
	if err != nil || !stat.Mode().IsRegular() || stat.Size() < 5 {
		return errors.New("no application default credentials found; please create them by running `gcloud auth application-default login`")
	}
	return nil
}

func hasGcloudConfig(configDir string) bool {
	if stat, err := os.Stat(path.Join(configDir, "configurations")); os.IsNotExist(err) || !stat.IsDir() {
		return false
	}
	if stat, err := os.Stat(path.Join(configDir, "active_config")); os.IsNotExist(err) || !stat.Mode().IsRegular() {
		return false
	}
	return true
}
