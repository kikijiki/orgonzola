{
  description = "orgonzola dev shell: stable Rust, Tauri v2 (WebKitGTK), Node and pnpm";

  inputs = {
    # Tracks the host's channel so dev shell libraries (fontconfig included) come from the same cache.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Toolchain comes from ./rust-toolchain.toml.
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # Tauri v2 Linux deps (WebKitGTK + GTK).
        tauriLibs = with pkgs; [
          webkitgtk_4_1
          gtk3
          glib
          libsoup_3
          gdk-pixbuf
          cairo
          pango
          harfbuzz
          librsvg
          at-spi2-atk
          dbus
          openssl
        ];
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkg-config
            gobject-introspection
            rustToolchain
            cargo-tauri

            nodejs_22
            pnpm

            just
            tokei

            prettier      # `just docs-fix`
            lychee        # `just docs-check`

            # llama-cpp-sys-2 builds the vendored llama.cpp with cmake and bindgen (libclang).
            # The Vulkan backend needs glslc, which shaderc provides.
            cmake
            shaderc
          ];

          buildInputs = tauriLibs ++ (with pkgs; [
            # Vulkan headers and loader. The GPU driver comes from the host at runtime.
            vulkan-headers
            vulkan-loader
          ]);

          shellHook = ''
            # WebKitGTK paints a blank window on some GPU drivers with the default compositing path.
            export WEBKIT_DISABLE_COMPOSITING_MODE=1
            export WEBKIT_DISABLE_DMABUF_RENDERER=1

            # GTK needs its gsettings schemas from the nix store, or file dialogs warn or crash.
            export XDG_DATA_DIRS="${pkgs.gtk3}/share/gsettings-schemas/${pkgs.gtk3.name}:${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/${pkgs.gsettings-desktop-schemas.name}''${XDG_DATA_DIRS:+:$XDG_DATA_DIRS}"

            # On nix, libclang does not find glibc's headers. Pass it the cc-wrapper's include flags
            # plus clang's builtin-header dir so bindgen can parse llama.cpp.
            export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"
            export BINDGEN_EXTRA_CLANG_ARGS="$(< ${pkgs.stdenv.cc}/nix-support/libc-crt1-cflags) $(< ${pkgs.stdenv.cc}/nix-support/libc-cflags) $(< ${pkgs.stdenv.cc}/nix-support/cc-cflags) -idirafter ${pkgs.libclang.lib}/lib/clang/${pkgs.lib.versions.major pkgs.libclang.version}/include"
            # Keep libvulkan on the loader path so `cargo tauri dev` can resolve it.
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            echo "orgonzola devshell"
            echo "  rustc:       $(rustc --version 2>/dev/null)"
            echo "  cargo-tauri: $(cargo tauri --version 2>/dev/null || echo '<run inside a workspace>')"
            echo "  node:        $(node --version 2>/dev/null)"
            echo "  pnpm:        $(pnpm --version 2>/dev/null)"
          '';
        };
      });
}
