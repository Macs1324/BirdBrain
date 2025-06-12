{
  inputs = {
    nixpkgs = {
      url = "github:NixOS/nixpkgs/nixos-unstable";
    };
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
  };
  outputs =
    { self
    , nixpkgs
    , flake-utils
    , rust-overlay
    ,
    }:
    flake-utils.lib.eachDefaultSystem
      (
        system:
        let
          overlays = [
            (import rust-overlay)
          ];
          pkgs = import nixpkgs {
            inherit system overlays;
            config.allowUnfree = true;
          };
          rustToolchain = with pkgs; [ rust-bin.stable.latest.default ];

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustToolchain
            git
            clang
            llvmPackages.libclang
          ];

          buildInputs = with pkgs; [
            cmake
            udev
            ninja
            openssl
            pango
            gdk-pixbuf
            gtk3
            libdrm
            mesa
            wlroots
            libopus
            alsa-lib
            vulkan-loader
            systemd
            libxkbcommon
            xorg.libX11
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXi
            xorg.libxcb
            xorg.xkbcomp
            xorg.xkbutils
            xorg.xkbevd
            libGL
            wayland
            # Webcam support
            v4l-utils
            libv4l
            libudev-zero
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-bad
            gst_all_1.gst-plugins-ugly
            stdenv.cc.cc.lib
            libz
            python313Packages.pillow
          ];
        in
        with pkgs; {
          devShells.default = mkShell.override { stdenv = pkgs.clang17Stdenv; } {
            inherit buildInputs nativeBuildInputs;
            shellHook = ''
              export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib.makeLibraryPath buildInputs}
              export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
            '';
          };
        }
      );
}
